import torch
import torch.autograd as autograd
import os, sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTv import MoreauProximator
from utilsInfConvTraining import TV_Solver_Training, K_fixedPoint, JacobianFixedPointP, fixedPointP

lmbd = 1e-2
bounds = [None, None]
img = torch.rand(1,1,40,40) + torch.full((1,1,40,40), 1.)

z, P = TV_Solver_Training(img, lmbd, batch=False, enforce_positivity = True)

hat_P = fixedPointP(P, img, lmbd, tau=1/8, batch = False, device = "cpu")

jacobian = autograd.functional.jacobian(lambda x: fixedPointP(x.view_as(P),img,lmbd=lmbd, tau = 1/8., batch = False, device = "cpu").view(-1), P.view(-1))


jacobianManual = JacobianFixedPointP(P, img, 8., lmbd, "cpu")

print(torch.sqrt(torch.sum(torch.pow(hat_P, 2))))
print(torch.isnan(jacobianManual).any())
print(torch.isnan(jacobian).any())
print(torch.sqrt(torch.sum(torch.pow(jacobianManual-jacobian, 2))))










