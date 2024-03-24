import torch
import sys
import os
import math
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tv_prox import CostTV

from utils import tStepDenoiser


def TV_Train_Recon(y, alpha, lmbd, H, Ht, tIter):
    x = torch.zeros_like(Ht(y))
    z = torch.clone(x)

    t = 1

    def grad_func(x):
        return alpha * Ht(H(x)-y)
    
    cost_tv = CostTV(x.squeeze().shape, lmbd, device = x.device)

    for i in range(tIter):
        x_old = torch.clone(x)
        x = z - grad_func(z)

        x = cost_tv.applyProx(x, alpha)
        
        t_old = t
        t = 0.5 *(1 + math.sqrt(1 + 4*t**2))
        
        z = x + (t_old -1)/t *(x - x_old)

    return x

def step_InfConv_Denoiser(model, x_noisy, lmbdLagrange, alpha, tIter, tIterCRRNN, tIterTV):

    u_k_1 = torch.clone(x_noisy)

    data_b = torch.clone(x_noisy)

    z = torch.clone(x_noisy)

    w = torch.clone(x_noisy)

    g = torch.clone(x_noisy)

    theta1 = torch.clone(x_noisy)

    theta2 = torch.clone(x_noisy)

    H = lambda x: x
    Ht = lambda x: x

    for i in range(tIter):
        ### u-update ###
        u_k_1 =  (1/(1+2*lmbdLagrange))* (x_noisy + lmbdLagrange*(z + w + theta1 + g - theta2))
        
        ### z-update ###
        data_b = torch.clone(u_k_1-w-theta1)
        z = TV_Train_Recon(data_b, alpha = 1, lmbd = alpha/lmbdLagrange, H = H, Ht = Ht, tIter = tIterTV)

        ### w-update ###
        data_b = torch.clone()
        w = tStepDenoiser(model = model, x_noisy = data_b, t_steps = tIterCRRNN)

        ### g-update ###
        data_b = torch.clone(u_k_1 + theta2)
        g = torch.clip(data_b, 0, None)

        ### theta-update ###
        theta1 = theta1 - u_k_1 + z + w
        theta2 = theta2 + u_k_1 - g

    return u_k_1


    pass
