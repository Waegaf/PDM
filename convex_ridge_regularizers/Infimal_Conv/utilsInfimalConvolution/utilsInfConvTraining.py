import torch
import sys
import os
import math
from tqdm import tqdm
import torch.autograd as autograd
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTv import TV_reconstruction, Tv_denoising_reconstruction, MoreauProximator
from reconstruction_map_crr import AdaGD_Recon, AdaAGD_Recon




def TV_Solver_Training(y, lmbd):
    bounds = [None, None]
    moreauProx = MoreauProximator(y.squeeze.shape(), lmbd, bounds, device = y.device)
    z, P = moreauProx.applyProxPrimalDual(y, alpha = 1.0)
    return z, P


def CRR_NN_Solver_Training(y, model, lmbd = 1, mu = 1, max_iter = 200):
    """"This solver uses the adaptive gradient descent scheme """

    def grad_func(x):
        return ((x-y) + lmbd*model(mu*x))
    
    # Initialization
    tol = 1e-6
    alpha = 1e-5
    beta = 1e-5
    x_old = torch.zeros_like(y)
    grad = grad_func(x_old)

    x = x_old - alpha*grad
    z = torch.clone(x)
    theta = float('inf')
    Theta = float('inf')

    for t in range(max_iter):
        grad_old = torch.clone(grad)
        grad = grad_func(x)

        alpha_1 = (torch.norm(x-x_old)/torch.norm(grad - grad_old)).item()/2
        alpha_2 = math.sqrt(1 + theta/2) * alpha

        alpha_old = alpha
        alpha = min(alpha_1, alpha_2)

        beta_1 = 1 / 4 / alpha_1
        beta_2 = math.sqrt(1 + Theta/2) * beta

        beta_old = min(beta_1, beta_2)

        gamma = (1/math.sqrt(alpha) - math.sqrt(beta)) / (1/math.sqrt(alpha) + math.sqrt(beta))

        z_old = torch.clone(z)

        z = x -alpha*grad
        x_old = torch.clone(x)

        x = z + gamma * (z- z_old)

        theta = alpha / alpha_old
        Theta = beta / beta_old

        relNorm = (torch.norm(x_old - x)/torch.norm(x_old)).item()

        if relNorm < tol:
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


def H_fixedPoint(x):
    pass



def tstepInfConvDenoiser(model, x_noisy, t_steps, alpha, beta, **kwargs):
    lmbdLagrange = kwargs.get('lmbdLagrange', 1e-2)


    # learnable regularization parameters
    lmbd = model.lmbd_transformed
    mu = model.mu_transformed

    # Lipschitz bound of the model estimated in a differentiable way
    if model.training:
        L = torch.clip(model.precise_lipschitz_bound(n_iter = 2, differentiable = True,), 0.1, None)

        model.L.data = L
    else:
        L = model.L

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
        u = (1/(2*lmbdLagrange))*(x_noisy + lmbdLagrange*(z + w + Theta1 + g -Theta2))

        # z-update
        with torch.no_grad():
            data_z = u - w - Theta1
            z, P = TV_Solver_Training(data_z, lmbd = alpha/lmbdLagrange)
        y = vectorize(z, P)
        y = y - K_fixedPoint(z, P, data_z, lmbdLagrange = lmbdLagrange, alpha = alpha)
        jacobianY = autograd.functional.jacobian(lambda x: K_fixedPoint(*matricize(x), data_z))
        y.register_hook(lambda grad: torch.linalg.solve(jacobianY.transpose(), grad))
        z, P = matricize(y)

        # w-update
        with torch.no_grad():
             data_w = u - z -Theta1
             w = CRR_NN_Solver_Training(y = data_w, H=H, Ht = Ht, model = model, lmbd = 2*beta/lmbdLagrange , mu = 1 ,tol=1e-6, max_iter = 100, enforce_positivity = False )
            
        w = w - H_fixedPoint(w, data_w)
        jacobianW = autograd.function.jacobian(lambda x: H_fixedPoint(x, data_w))
        w.register_hook(lambda grad: torch.linalg.solve(jacobianW.transpose(), grad))

        # g-update
        g = torch.clip(u + Theta2, 0, None)
        Theta1 = Theta1 -u + z + w
        Theta2 = Theta2 + u - g
    
    return u