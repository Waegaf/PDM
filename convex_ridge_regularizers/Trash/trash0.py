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

import sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTv import MoreauProximator

device = "cpu"

imgLenna = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Lenna.png", cv2.IMREAD_GRAYSCALE), (100, 100))
imgLenna_torch = torch.tensor(imgLenna, device = device).reshape((1,1) + imgLenna.shape)/255
imgLenna_torchNoisy = imgLenna_torch + 25/255 * torch.randn_like(imgLenna_torch)

imgMountain = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Mountain.jpg", cv2.IMREAD_GRAYSCALE),(100,100))
imgMountain_torch = torch.tensor(imgMountain, device = device).reshape((1,1)+ imgMountain.shape)/255
imgMountain_torchNoisy = imgMountain_torch + 25/255 * torch.randn_like(imgMountain_torch)

batchImg = torch.empty(2,1,100,100)
batchImg[0,...] = imgLenna_torchNoisy
batchImg[1,...] = imgMountain_torchNoisy


lmbd = 5e-2
bounds = [None, None]
device = "cpu"
batch = True
Prox = MoreauProximator([100,100],lmbd, bounds = bounds, device = device, batch = batch)

res = Prox.batch_applyProx(batchImg, alpha = 1.0)

print(ssim(imgLenna_torch, res[0,...].unsqueeze(0), data_range=1.))

