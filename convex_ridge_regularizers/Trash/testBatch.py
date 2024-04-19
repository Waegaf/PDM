import numpy as np
import argparse
import json
import torch
import torch.autograd as autograd
from PIL import Image
import math
import cv2
import torch.nn as nn
from functools import partial
import pylops_gpu
from torchmetrics.functional import structural_similarity_index_measure as ssim
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
from utilsInfConvTraining import TV_Solver_Training, CRR_NN_Solver_Training
from models import utils


torch.manual_seed(561)
device = "cpu"
sigma_training = 5
t = 10
exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)

imgLenna = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Lenna.png", cv2.IMREAD_GRAYSCALE), (400, 400))
imgLenna_torch = torch.tensor(imgLenna, device = device).reshape((1,1) + imgLenna.shape)/255
imgLenna_torchNoisy = imgLenna_torch + 25/255 * torch.randn_like(imgLenna_torch)

imgMountain = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Mountain.jpg", cv2.IMREAD_GRAYSCALE),(400,400))
imgMountain_torch = torch.tensor(imgMountain, device = device).reshape((1,1)+ imgMountain.shape)/255
imgMountain_torchNoisy = imgMountain_torch + 25/255 * torch.randn_like(imgMountain_torch)

batchImg = torch.empty(2,1,400,400)
batchImg[0,...] = imgLenna_torchNoisy
batchImg[1,...] = imgMountain_torchNoisy


batchTV, batchP = TV_Solver_Training(batchImg, 5e-2, True, True)
batchCRRNN = CRR_NN_Solver_Training(batchImg, model, lmbd = 25)

fig = plt.figure()
gs = GridSpec(nrows= 2, ncols = 3)

ax0 = fig.add_subplot(gs[0,0])
ax0.set_title("Clean Image")
ax0.imshow(imgLenna_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax0.set_yticks([])
ax0.set_xticks([])

ax0 = fig.add_subplot(gs[1,0])
ax0.set_title("Clean Image")
ax0.imshow(imgMountain_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax0.set_yticks([])
ax0.set_xticks([])

ax0 = fig.add_subplot(gs[0,1])
ax0.set_title("Clean Image")
ax0.imshow(batchTV[0,...].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax0.set_yticks([])
ax0.set_xticks([])


ax0 = fig.add_subplot(gs[1,1])
ax0.set_title("Clean Image")
ax0.imshow(batchTV[1,...].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax0.set_yticks([])
ax0.set_xticks([])

ax0 = fig.add_subplot(gs[0,2])
ax0.set_title("Clean Image")
ax0.imshow(batchCRRNN[0,...].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax0.set_yticks([])
ax0.set_xticks([])

ax0 = fig.add_subplot(gs[1,2])
ax0.set_title("Clean Image")
ax0.imshow(batchCRRNN[1,...].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax0.set_yticks([])
ax0.set_xticks([])

plt.show()
