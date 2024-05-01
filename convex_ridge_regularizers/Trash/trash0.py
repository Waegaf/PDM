import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.gridspec import GridSpec
import torch.autograd as autograd
# Add of the corresponding paths to be enable to use the wanted functions
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
from models import utils
from reconstruction_map_crr import AdaGD_Recon, AdaAGD_Recon
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
# sys.path.append("/cs/research/vision/home2/wgafaiti/Code/convex_ridge_regularizers/Infimal_conv/utilsInfimalConvolution")
from utilsInfConvTraining import CRR_NN_Solver_Training, H_fixedPoint

# Creation of a folder to save the results
if not os.path.exists("Infimal_Conv/CRRNN/ResultsCRRNN"):
    os.makedirs("Infimal_Conv/CRRNN/ResultsCRRNN")

# Set seed for the noise added
torch.manual_seed(61)




def f():
    return 2
    print()