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
from utilsTv import TV_reconstruction, Tv_denoising_reconstruction, MoreauProximator, LinearOperator, LinearOperatorBatch
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

def K_fixedPoint(z, P, data, lmbd, tau = 1e-2, batch = True, device = "cpu"):
    
    sizein = [z.shape[2], z.shape[3]]
    if batch:
        LinearOp = LinearOperatorBatch(sizein=sizein, device= device)
        gradZ = LinearOp.batch_applyL_t(z)
        minusDivP = LinearOp.batch_applyL(P)
    else:
        LinearOp = LinearOperator(sizein= sizein, device = device)
        gradZ = LinearOp.applyL_t(z)
        minusDivP = LinearOp.applyL(P)

    sigma = 1/tau
    def I_dF_1(P_like):
        n = P_like.size(dim = 1)
        m = P_like.size(dim = 2)
        if batch:
            denom = torch.clamp(torch.sqrt(torch.sum(torch.pow(P_like, 2), dim=1)), min=1.0).unsqueeze(1)
            P_tilde = P_like/denom.expand(-1, 2, -1, -1)
        else:
            denom = torch.clamp(torch.sqrt(torch.sum(torch.pow(P_like, 2), dim=0)), min=1.0)
            P_tilde = P_like/denom.expand(2, n, m)
        return P_tilde
    
    def I_dG_1(z_like):
        z_tilde = (z_like + (tau*lmbd)*data)/(1 + (tau*lmbd))
        return z_tilde

    fixedP = I_dF_1(P+sigma*gradZ)-P
    fixedZ = I_dG_1(z+tau*minusDivP)-z

    return fixedZ, fixedP

def fixedPointP(P, g, lmbd, tau, batch = False, device = "cpu"):
    sigma = 1/tau
    if batch:
        sizein = [P.shape[2], P.shape[3]]
        LinearOp = LinearOperatorBatch(sizein=sizein, device= device)
        minusDivP = LinearOp.batch_applyL(P)
        interRes0 = LinearOp.batch_applyL_t(g-lmbd*minusDivP)
        interRes2 = P+sigma*interRes0
        norm = torch.clamp(torch.sqrt(torch.sum(torch.pow(interRes2, 2), dim=1)), min=1.0).unsqueeze(1)
        interRes3 = interRes2/norm.expand(-1, 2, -1, -1)
        return interRes3-P 
    else:
        sizein = [P.shape[1], P.shape[2]]
        LinearOp = LinearOperator(sizein= sizein, device = device)
        minusDivP = LinearOp.applyL(P)
        interRes0 = LinearOp.applyL_t(g-lmbd*minusDivP)
        interRes2 = P+sigma*interRes0
        norm = torch.clamp(torch.sqrt(torch.sum(torch.pow(interRes2, 2), dim=0)), min=1.0)
        interRes3 = interRes2/norm.expand(2, sizein[0], sizein[1])
        return interRes3-P 
    


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


def JacobianProjUnitBall(P):
    dim = P.shape[1]*P.shape[2]
    PFlatten = torch.cat((P[0,...].view(1,-1), P[1,...].view(1,-1)),0)
    norm = torch.sum(torch.pow(P, 2), dim=0)
    normFlatten = norm.view(-1)
    is_norm_larger_than_1 = normFlatten > 1.
    is_norm_larger_than_1_2 = torch.cat((is_norm_larger_than_1, is_norm_larger_than_1), 0)[None,:]

    
    

    IndicesUP = -(torch.arange(dim)[:, None] +1.0) + ((torch.arange(2*dim)[None, :] +1.0))
    boolean = (IndicesUP == dim) | (IndicesUP == 0)
    backgroundUp = torch.where(boolean, torch.tensor(1.0), torch.tensor(0.0))
    
    background = torch.cat((backgroundUp, backgroundUp), 0)

    productA = (background * is_norm_larger_than_1_2) == 1

    def derivative_i_equal_j(i, k):  
        norm =  torch.pow(normFlatten[i-(1-k)*dim], 3/2)
        return PFlatten[k,i]/norm
        
    def derivative_i_not_equal_j(i,k):
        return -PFlatten[1,i]*PFlatten[0,i]/torch.pow(normFlatten[i-(1-k)*dim], 3/2)

    ID = torch.eye(2*dim)
    IDMod = ID * torch.cat((torch.arange(dim)[:,None], torch.arange(dim)[:,None]), 0)
    IDMod = IDMod.to(torch.int32)
    K = ID * (torch.cat((torch.full((dim,1), 1), torch.full((dim,1), 0)), 0))
    K = K.to(torch.int32)
    newProdA = productA.long()*ID
    newProdA = newProdA == 1
    productB0 = torch.where(newProdA, derivative_i_equal_j(IDMod, K), torch.zeros_like(productA) )

    for i in range(2*dim):
        for j in range(2*dim):
            if productB0[i,j].isnan().item():
                print(f"({i}, {j})")
                print(IDMod[i,j])

    IDMod = (productA.long()*(torch.ones_like(ID)- ID))* torch.cat((torch.arange(dim)[:,None], torch.arange(dim)[:,None]), 0)
    IDMod = IDMod.to(torch.int32)
    prodB1_bool = (productA.long()-ID)==1
    productB1 = torch.where(prodB1_bool, derivative_i_not_equal_j(IDMod,K), torch.zeros_like(productA) )
    productB = productB0+productB1
    productA = productA.long()
    print(f"PRODUCTB0 HAS A NAN:{torch.isnan(productB0).any()}")
    print(f"PRODUCTB1 HAS A NAN:{torch.isnan(productB1).any()}")
    return background + (productA*productB)

def JacobianDivergence(n,m):
    
    def derivateP_1(i,j):
        leftMatrix = torch.zeros(n,m)
        leftMatrix[i,j] = 1.0
        if i+1 < n:
            leftMatrix[i+1,j] = -1.
        return leftMatrix.view(-1)
    
    def derivateP_2(i,j):
        rigthMatrix = torch.zeros(n,m)
        rigthMatrix[i,j] = 1.0
        if j+1 < m:
            rigthMatrix[i,j+1] = -1.0
        return rigthMatrix.view(-1)

    for i in range(n):
        for j in range(m):
            if i==0 and j==0:
                rigthMatrix = derivateP_2(i,j)[None, :]
                leftMatrix = derivateP_1(i,j)[None,:]
            else:
                rigthMatrix = torch.cat((rigthMatrix, derivateP_2(i,j)[None, :]), 0)
                leftMatrix = torch.cat((leftMatrix, derivateP_1(i,j)[None, :]), 0)

    
    JacobianDiv = torch.cat((leftMatrix, rigthMatrix), 1)
    return JacobianDiv

def JacobianGrad(n,m):
    
    def derivateGrad_1(i,j):
        rigthMatrix = torch.zeros(n,m)
        if i < n-1:
            rigthMatrix[i,j] = -1.0
            rigthMatrix[i+1,j] = 1.0
        return rigthMatrix.view(-1)
    
    def derivateGrad_2(i,j):
        leftMatrix = torch.zeros(n,m)
        if j < m-1:
            leftMatrix[i,j] = -1.0
            leftMatrix[i,j+1] = 1.0
        return leftMatrix.view(-1)
    
    for i in range(n):
        for j in range(m):
            if i==0 and j==0:
                rigthMatrix = derivateGrad_1(i,j)[None, :]
                leftMatrix = derivateGrad_2(i,j)[None,:]
            else:
                rigthMatrix = torch.cat((rigthMatrix, derivateGrad_1(i,j)[None, :]), 0)
                leftMatrix = torch.cat((leftMatrix, derivateGrad_2(i,j)[None, :]), 0)
    

    JacobianGrad = torch.cat((rigthMatrix, leftMatrix), 0)
    return JacobianGrad

def JacobianFixedPointP(P, img, sigma, lmbd, device):
    n = P.shape[1]
    m = P.shape[2]

    LinearOp = LinearOperator([n,m], device = device)
    divP = -LinearOp.applyL(P)

    z = img + lmbd*divP
    gradZ = LinearOp.applyL_t(z)
    J0 = JacobianProjUnitBall(P+sigma*gradZ)
    J1 = JacobianDivergence(n,m)
    J2 = JacobianGrad(n,m)
    Id = torch.eye(n*m*2)
    if torch.isnan(J0).any():
        print("STOOOOOOp")
    B1 = Id + torch.matmul(J2,J1)
    B0 = torch.matmul(J0, B1)
    
    return B0 - Id
