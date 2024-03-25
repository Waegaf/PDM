import torch
import sys
import os
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
sys.path.append('C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/others/tv')
from tv_prox import CostTV
from utilsTv import TV_reconstruction, Tv_denoising_reconstruction
from reconstruction_map_crr import AdaGD_Recon, AdaAGD_Recon





def reconstruction_map_InfTVCRRNN(model, x_noisy, lmbdLagrange, alpha, beta, maxIter, x_origin = None, **kwargs):
    ### Initialization of the variable of the ADMM scheme ###
    maxIterTVRecon = kwargs.get('maxIterTVRecon', 200)
    maxIterCRRNNRecon = kwargs.get('maxIterCRRNNRecon', 200)
    trackCost = kwargs.get('trackCost', False)
    stopTol = kwargs.get('stopTol', 1e-4)

    if trackCost:
        regCostList = []
        psnrList = []
        ssimList = []


    u_k = torch.clone(x_noisy)
    # u_k = torch.ones_like(x_noisy)
    z = torch.clone(u_k)
    w = torch.clone(u_k)
    g = torch.clone(u_k)

    theta1 = torch.clone(u_k)
    theta2 = torch.clone(u_k)

    ## Initialization of the parameters for CRRNN ##
    H = lambda x: x
    Ht = lambda x: x

    # with torch.no_grad() since we are in the final reconstruction phase
    with torch.no_grad():
        for i in tqdm(range(maxIter), desc = "Outer loop"):
            ### u-update ###
            u_k_1 = (1/(1+2*lmbdLagrange))* (x_noisy + lmbdLagrange*(z + w + theta1 + g - theta2))
            # relativeError = norm(u_k_1 - u_k) / norm(u_k_1)
            # tol = norm(u_k_1 - u_k)/norm(u_k_1);
            tol = torch.sum(torch.pow(u_k_1 - u_k, 2)).item() / torch.sum(torch.pow(u_k_1, 2)).item()


            ### z-update ###
            data_b = torch.clone(u_k_1 - w - theta1)
            z = Tv_denoising_reconstruction(data_b, alpha = 1, lmbd = alpha/lmbdLagrange,  x_init = data_b, enforce_positivity = False)

            ### w-update ###
            data_b = torch.clone(u_k_1 -z - theta1)
            w, psnrCRRNN, ssimCRRNN, nIterCRRNN = AdaAGD_Recon(y = data_b, H=H, Ht = Ht, model = model, lmbd = 2*beta/lmbdLagrange , mu = 1 ,tol=1e-6, max_iter = maxIterCRRNNRecon, enforce_positivity = False )

            ### g-update ###
            data_b = torch.clone(u_k_1 + theta2)
            g = torch.clip(data_b, 0, None)

            ### theta-update ###
            theta1 = theta1 - u_k_1 + z + w
            theta2 = theta2 + u_k_1 - g

            # update u (for the computation of tol)
            u_k = u_k_1
            
            if x_origin is not None:
                psnrImg = psnr(u_k, x_origin, data_range= 1.0).item()
                ssimImg = ssim(u_k, x_origin, data_range = 1.0).item()
                regCost = Regularization_cost_InfTVCRRNN(u_k, z, w, g, model, lmbdLagrange, alpha, beta)
            else:
                psnrImg = None
                ssimImg = None
                regCost = Regularization_cost_InfTVCRRNN(u_k, z, w, g, model, lmbdLagrange, alpha, beta)
            
            if trackCost:
                psnrList.append(psnrImg)
                ssimList.append(ssimImg)
                regCostList.append(regCost)

            if tol < stopTol:
                break

    if trackCost:
        return u_k, z, w, g, psnrList, ssimList, regCostList
    else:
        return u_k, z, w, g, psnrImg, ssimImg, regCost



def Regularization_cost_InfTVCRRNN(u, z, w, g, model, lmbdLagrange, alpha, beta):

    cost_tv = CostTV(z.squeeze().shape, alpha, device = 'cpu')
    costtv = cost_tv.apply(z).item()

    costCRRNN = (beta * model.cost(w)).item()

    g0 = torch.zeros_like(g)
    g0[g < 0] = 1
    costKsi = torch.count_nonzero(g0).item()

    firstConstraint = torch.sum(torch.pow(u - z - w, 2)).item()
    secondConstraint = torch.sum(torch.pow(u - g, 2)).item()


    return (costtv + costCRRNN + costKsi + lmbdLagrange*(firstConstraint + secondConstraint))



