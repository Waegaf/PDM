import torch
import sys
import os
import math
from tqdm import tqdm
# import torch.autograd as autograd
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTv import TV_reconstruction, Tv_denoising_reconstruction, MoreauProximator
# from reconstruction_map_crr import AdaGD_Recon, AdaAGD_Recon




def TV_Solver_Training(y, lmbd, batch, enforce_positivity):
    '''This solver only solves the TV problem in the denoising task, i.e. H = Id'''
    if enforce_positivity:
        bounds = [0.0, float('Inf')]
    else:
        bounds = [None, None]

    moreauProx = MoreauProximator([y.shape[2], y.shape[3]], lmbd, bounds, device = y.device, batch = batch)
    if not batch:
        z, P = moreauProx.applyProxPrimalDual(y, alpha = 1.0)
    else:
        z, P = moreauProx.batch_applyProxPrimalDual(y, alpha = 1.0)
    return z, P


def CRR_NN_Solver_Training(y, model, lmbd = 1, mu = 1, max_iter = 300, batch = True, enforce_positivity = False, device ="cpu"):
    """"This solver uses the adaptive gradient descent scheme """

    def grad_func(x):
        return ((x-y) + lmbd*model(mu*x))
    
    # Initialization
    nbatches = y.shape[0]
    tol = 1e-3
    alpha = torch.full((nbatches, 1),1e-5, device = device)
    beta = torch.full((nbatches, 1), 1e-5, device = device)
    x_old = torch.zeros_like(y, device = device)
    grad = grad_func(x_old)

    def prod(a, x):
        out = torch.empty_like(x, device= device)
        for i in range(nbatches):
            out[i,...] = a[i].item() * x[i,...]
        return out

    x = x_old - prod(alpha, grad)
    z = torch.clone(x)
    theta = torch.full((nbatches,1),float('inf'), device = device)
    Theta = torch.full((nbatches, 1), float('inf'), device = device)

    for t in range(max_iter):
        grad_old = torch.clone(grad)
        if torch.any(x.isnan()).item():
            print(torch.mean(grad_norm))
            print(x)
        grad = grad_func(x)

        normx = torch.norm(x-x_old, dim = (2,3))
        normgrad = torch.norm(grad - grad_old, dim = (2,3))
        diff = normx-normgrad
        alpha_1 = 0.5*(torch.clamp(torch.norm(x-x_old, dim = (2,3)), 1e-7, None))/torch.clamp(torch.norm(grad - grad_old, dim = (2,3)), 1e-7, None)
        alpha_2 = torch.sqrt(1 + theta/2) * alpha

        alpha_old = alpha
        alpha = torch.min(alpha_1, alpha_2)

        beta_1 = 1 / 4 / alpha_1
        beta_2 = torch.sqrt(1 + Theta/2) * beta

        beta_old = torch.min(beta_1, beta_2)

        gamma = (1/torch.sqrt(alpha) - torch.sqrt(beta)) / (1/torch.sqrt(alpha) + torch.sqrt(beta))

        z_old = torch.clone(z)

        z = x -prod(alpha,grad)
        x_old = torch.clone(x)

        x = z + prod(gamma ,(z- z_old))

        

        theta = alpha / alpha_old
        Theta = beta / beta_old


        if enforce_positivity:
            x = torch.clamp(x, 0, None)

        grad_norm = torch.norm(grad, dim = (2,3))
        if torch.mean(grad_norm).item() < tol:
            print("tol reached")
            break
        
    return x


def vectorize(z, P):
    return torch.cat([z.view(-1), P.view(-1)])

def matricize(M, z, P):
    zNew = M[:z.numel()].view_as(z)
    pNew = M[z.numel():].view_as(P)
    return zNew, pNew

def K_fixedPoint(z, P, data, lmbdLagrange, alpha, tau = 1e-2):
    
    def I_dF_1(P_like):
        n = P_like.size(dim = 1)
        m = P_like.size(dim = 2)
        denom = torch.clamp(torch.sqrt(torch.sum(torch.pow(P_like, 2), dim=0)), min=1.0)
        P_tilde = P_like/denom.expand(2, n, m)
        return P_tilde
    
    def I_dG_1(z_like):
        z_tilde = (z_like + (tau*lmbdLagrange/alpha)*data)/(1 + (tau*lmbdLagrange/alpha))
        return z_tilde

    K_1 = I_dF_1(P)-P
    K_2 = I_dG_1(z)-z

    return vectorize(K_1, K_2)


def H_fixedPoint(x, model, data, lmbdLagrange, beta, mu):
    return (x-data) + beta/lmbdLagrange*(model(mu*x))



def tstepInfConvDenoiser(model, x_noisy, t_steps, alpha,  **kwargs):
    lmbdLagrange = kwargs.get('lmbdLagrange', 1e-2)


    # learnable regularization parameters
    beta = model.lmbd_transformed
    mu = model.mu_transformed

    # Lipschitz bound of the model estimated in a differentiable way (We probably won't use it with AdaAGD_Recon. So, we should not use it)
    # if model.training:
    #     L = torch.clip(model.precise_lipschitz_bound(n_iter = 2, differentiable = True,), 0.1, None)

    #     model.L.data = L
    # else:
    #     L = model.L

    # Initialization of the tensors
    z = torch.zeros_like(x_noisy)
    w = torch.zeros_like(x_noisy)
    g = torch.zeros_like(x_noisy)
    Theta1 = torch.zeros_like(x_noisy)
    Theta2 = torch.zeros_like(x_noisy)

    # Initialization of the linear operator to identities (since we are in the denoising task)
    H = lambda x: x
    Ht = lambda x: x

    for t in range(t_steps):
        # Differentiable stpes
        # u-update
        u = (1/(1 + 2*lmbdLagrange))*(x_noisy + lmbdLagrange*(z + w + Theta1 + g -Theta2))

        # z-update
        with torch.no_grad():
            data_z = torch.clone(u - w - Theta1)
            z, P = TV_Solver_Training(data_z, lmbd = alpha/lmbdLagrange)
        y = vectorize(z, P)
        y = y - K_fixedPoint(z, P, data_z, lmbdLagrange = lmbdLagrange, alpha = alpha)
        jacobianY = autograd.functional.jacobian(lambda x: K_fixedPoint(*matricize(x, z, P), data = data_z, lmbdLagrange = lmbdLagrange, alpha = alpha), y)
        if y.requires_grad:
            y.register_hook(lambda grad: torch.linalg.solve(jacobianY.transpose(), grad))
        z, P = matricize(y, z, P)

        # w-update
        with torch.no_grad():
             data_w = u - z -Theta1
             w = CRR_NN_Solver_Training(y = data_w, model = model, lmbd = 2*beta/lmbdLagrange , mu = 1, max_iter = 200)
            
        y = w.view(-1)
        y1 = H_fixedPoint(w, model, data_w, lmbdLagrange = lmbdLagrange, beta = beta, mu = mu)
        y = y - y1.view(-1)
        jacobianW = autograd.functional.jacobian(lambda x: H_fixedPoint(x.view_as(w), model, data_w, lmbdLagrange = lmbdLagrange, beta = beta, mu = mu), y)
        if w.requires_grad:
            w.register_hook(lambda grad: torch.linalg.solve(jacobianW.transpose(), grad))

        # g-update
        g = torch.clip(u + Theta2, 0, None)
        Theta1 = Theta1 -u + z + w
        Theta2 = Theta2 + u - g
    
    return u