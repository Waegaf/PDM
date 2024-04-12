import torch
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import sys
import os
import pylops_gpu

# Projection function to the box(xmin,xmax), i.e. ensure that xmin <= x <= xmax.
def projectionBox(x, xmin, xmax):
    # If there are no constraint, we simply return x
    if xmin is None and xmax is None:
        return x
    out = torch.clamp(x, min = xmin, max = xmax)
    return out


# The Linear operator corresponds to L (Beck and Teboulle, 2008)
class LinearOperator:

    def __init__(self, sizein, device) -> None:
        # Attributes
        # Number of samples in direction 0
        self.height = sizein[0]
        # Number of samples in direction 1
        self.width = sizein[1]
        # Device to work on (available since we work with pylops-gpu)
        self.device = device

        self.L_0 = pylops_gpu.FirstDerivative(N = self.height*self.width, dims = (self.height, self.width), dir = 0, device = self.device, togpu= (True, True))
        # Modification of the matrix of convolution H which is, by default, set to [0.5, 0, -0.5]. We need [1, -1, 0] (or [-1, 1, 0])
        self.L_0.Op.h[0,0,0] = 1
        self.L_0.Op.h[0,0,1] = -1
        self.L_0.Op.h[0,0,2] = 0

        self.L_1 = pylops_gpu.FirstDerivative(N = self.height*self.width, dims = (self.height, self.width), dir = 1, device = self.device, togpu= (True, True))
        self.L_1.Op.h[0,0,0] = 1
        self.L_1.Op.h[0,0,1] = -1
        self.L_1.Op.h[0,0,2] = 0

    # Implementation of the application of the ajdoint of L, L^t (i.e. the gradient)
    def applyL_t(self, x):
        x = x.to(torch.float)
        x.to(self.device)
        # The operators have to be applied on a flatten version of x
        out_0 = self.L_0*(x.reshape(-1))
        out_1 = self.L_1*(x.reshape(-1))
        out = torch.cat([torch.reshape(out_0, (1, self.height, self.width)), torch.reshape(out_1, (1, self.height, self.width))], dim = 0)
        return out # 2 x ... x ....
    
    # Implementation of the application of L (i.e -div)
    def applyL(self, y):
        y = y.to(self.device)
        out = torch.reshape(self.L_0.H*(y[0,...].view(-1)), (self.height, self.width)) + torch.reshape(self.L_1.H*(y[1,...].view(-1)), (self.height, self.width))
        out = out.unsqueeze(0)
        out = out.unsqueeze(0)
        return out # 1 x 1 x K x K





# Implemantaion of the proximal map of Moreau (Beck and Teboulle, 2008), that we called MoreauProximator
class MoreauProximator:

    def __init__(self, sizein, lmbd, bounds, device):
        
        # Attributes
        # Dimension of the image ([height, width])
        self.sizein = sizein
        # Regularizer paramater (fidelity_term + lambda*regularizer)
        self.lmbd = lmbd
        # Device to work on
        self.device = device
        # Bounds [l,u] of the box (Beck and Teboulle, 2008)
        self.bounds = bounds
        # LinearOperator L
        self.L = LinearOperator(sizein, device)

        # Parameters related to the FGP algorithm
        self.gamma = 1.0/8
        self.num_iter = 100

    
    # Implementation of the FGP algorithm (Beck and Teboulle, 2008)
    def applyProx(self, u, alpha):
        # Initialization phase
        alpha = alpha * self.lmbd
        P = torch.zeros(2, self.sizein[0], self.sizein[1], device = u.device)
        F = torch.zeros(2, self.sizein[0], self.sizein[1], device = u.device)
        t = 1.0

        # Begin of the iterations
        for iteration in range(self.num_iter):
            Pnew = F + (self.gamma/alpha)*(self.L.applyL_t(projectionBox(u - alpha*self.L.applyL(F),self.bounds[0], self.bounds[1])))
            tmp = torch.clamp(torch.sqrt(torch.sum(torch.pow(Pnew, 2), dim=0)), min=1.0)
            Pnew = Pnew/tmp.expand(2, self.sizein[0], self.sizein[1])
            tnew = (1 + math.sqrt(1 + 4*(t**2)))/2
            F = Pnew + (t-1)/tnew*(Pnew -P)
            t = tnew
            P = Pnew
            

        return projectionBox(u-alpha*self.L.applyL(P), self.bounds[0], self.bounds[1])
    
    def applyProxPrimalDual(self,u, alpha):
        # Initialization phase
        alpha = alpha * self.lmbd
        P = torch.zeros(2, self.sizein[0], self.sizein[1], device = u.device)
        F = torch.zeros(2, self.sizein[0], self.sizein[1], device = u.device)
        t = 1.0

        # Begin of the iterations
        for iteration in range(self.num_iter):
            Pnew = F + (self.gamma/alpha)*(self.L.applyL_t(projectionBox(u - alpha*self.L.applyL(F),self.bounds[0], self.bounds[1])))
            tmp = torch.clamp(torch.sqrt(torch.sum(torch.pow(Pnew, 2), dim=0)), min=1.0)
            Pnew = Pnew/tmp.expand(2, self.sizein[0], self.sizein[1])
            tnew = (1 + math.sqrt(1 + 4*(t**2)))/2
            F = Pnew + (t-1)/tnew*(Pnew -P)
            t = tnew
            P = Pnew
            

        return (projectionBox(u-alpha*self.L.applyL(P), self.bounds[0], self.bounds[1]), P)

        


# Implementation of the FISTA algorithm within the following problem troubleshooting:
    "argmin_x || H(x) - y||^2 + lambda*TV(x)"
def TV_reconstruction(y, alpha, lmbd, H, Ht, x_gt, **kwargs):

    # Keep the track the error w.r.t. to psnr and ssim metrics
    psnrList = []
    ssimList = []


    max_iter = kwargs.get('max_iter', 3000)
    tol = kwargs.get('tol', 1e-6)
    x_init = kwargs.get('x_init', None)
    enforce_positivity = kwargs.get('enforce_positivity', True)

    # Initialization phase
    if x_init is None:
        x = torch.zeros_like(Ht(y))
    else:
        x = x_init.clone()

    if enforce_positivity:
        bounds = [0.0, float('Inf')]
    else:
        bounds = [None, None]
    
    z = x.clone()
    t = 1
    def grad_func(x):
        return alpha * (Ht(H(x)-y))
    
    moreauProx = MoreauProximator(y.squeeze().shape, lmbd, bounds, device = y.device)

    for i in range(max_iter):
        x_old = torch.clone(x)
        x = z - grad_func(z)
        x = moreauProx.applyProx(x, alpha)
        t_old = t
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
        z = x + (t_old - 1)/t * (x - x_old)

        # Computation of the stopping criteria
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

        if x_gt is not None:
            psnrIteration = psnr(x, x_gt, data_range=1.0)
            ssimIteration = ssim(x, x_gt, data_range=1.0)

            psnrList.append(psnrIteration)
            ssimList.append(ssimIteration)
        if res < tol:
            break

        return (x, psnrList, ssimList, i)

# Implementaion of the simple denoising algorithm, i.e. H: x-> x, by only applying the 
# the proximal map of Moreau.
def Tv_denoising_reconstruction(y, lmbd, **kwargs):

    enforce_positivity = kwargs.get('enforce_positivity', True)

    if enforce_positivity:
        bounds = [0.0, float('Inf')]
    else:
        bounds = [None, None]

    moreauProx = MoreauProximator(y.squeeze().shape, lmbd, bounds, device = y.device)

    xDenoised = moreauProx.applyProx(y, alpha = 1)

    return xDenoised



