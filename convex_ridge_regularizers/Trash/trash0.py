import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.gridspec import GridSpec
# Add of the corresponding paths to be enable to use the wanted functions
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
from models import utils
from reconstruction_map_crr import AdaGD_Recon, AdaAGD_Recon

# Creation of a folder to save the results
if not os.path.exists("Infimal_Conv/CRRNN/ResultsCRRNN"):
    os.makedirs("Infimal_Conv/CRRNN/ResultsCRRNN")

# Set seed for the noise added
torch.manual_seed(61)

def kronecker_prod(W, A):
    p = W.shape[0]
    output = torch.empty_like(W)
    for i in range(p):
        output[i,:,:,:] = W[i,:,:,:] * A
    return output



# Choice of the (CRRNN) model
device = 'cpu'
sigma_training = 25
t = 10
exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)
# img = torch.randn(1,1,40,40)

# diff = model.get_derivative(img)
# print(diff.shape)





conv = model.conv_layer

unit = torch.empty((1600,1,40,40))
for i in range(1600):
    a = torch.zeros(1600)
    a[i] = 1.
    unit[i,:,:,:] = a.view(40,40)


result = conv(unit)

print(result.shape)



