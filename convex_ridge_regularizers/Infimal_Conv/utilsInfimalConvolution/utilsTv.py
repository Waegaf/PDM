import torch
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import sys
import os


class MoreauProximator:
    def __init__(self, sizein, lmbd, bounds, device):
        
        # Attributes
        self.sizein = sizein
        self.lmbd = lmbd
        self.device = device
        self.bounds = bounds

        # Parameters related to the FGP algorithm
        self.gamma = 1.0/8
        self.num_iter = 100

    

    def applyProx(self, u, alpha):
        
        alpha = alpha * self.lmbd
        P = torch.zeros(2, self.sizein[0], self.sizein[1], device = u.device)
        F = torch.zeros(2, self.sizein[0], self.sizein[1], device = u.device)

        t = 1.0

        for iteration in range(self.num_iter):
            # Pnew = ...
            tnew = (1 + math.sqrt(1 + 4*(t**2)))/2
            F = Pnew + (t-1)/tnew*(Pnew -P)
            t = tnew
            P = Pnew
            pass

        # return enforce_box_constraints(u - alpha)
        pass


def TV_reconstruction(y, alpha, lmbd, H, Ht, x_gt, **kwargs):
    psnrList = []
    ssimList = []


    max_iter = kwargs.get('max_iter', 3000)
    tol = kwargs.get('tol', 1e-6)
    x_init = kwargs.get('x_init', None)

    if x_init is None:
        x = torch.zeros_like(Ht(y))
    else:
        x = x_init.clone()
    
    z = x.clone()
    
    t = 1

    def grad_func(x):
        return alpha * (Ht(H(x)-y))
    
    for i in range(max_iter):
        x_old = torch.clone(x)
        x = z - grad_func(z)

        x = moreauProx.applyProx(x, alpha)

        t_old = t
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
        z = x + (t_old - 1)/t * (x - x_old)


        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()

        if x_gt is not None:
            psnrIteration = psnr(x, x_gt, data_range=1.0)
            ssimIteration = ssim(x, x_gt, data_range=1.0)

            psnrList.append(psnrIteration)
            ssimList.append(ssimIteration)
        if res < tol:
            break

        return (x, psnrList, ssimList, i)
