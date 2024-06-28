import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.gridspec import GridSpec
import json
from torch.utils.data import DataLoader
# Add of the corresponding paths to be enable to use the wanted functions
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
from models import utils
from training.data import dataset
from reconstruction_map_crr import AdaGD_Recon, AdaAGD_Recon
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics import StructuralSimilarityIndexMeasure as ssim



# Creation of a folder to save the results
if not os.path.exists("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_CRRNN_Sigma_5"):
    os.makedirs("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_CRRNN_Sigma_5")

# Set seed for the noise added
torch.manual_seed(61)

# Choice of the (CRRNN) model
device = 'cpu'

# Load model
sigma_training = 5
t = 5
exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)

# We need the identity operator and its adjoint (also identity) since we need to specify them to solve the denoising task with the CRRNN
H = lambda x: x
Ht = lambda x: x
# Upperbound of the Lip. Const. of the gradient of || H(x) - y ||^2, which is 1 in the identity case
opnorm = 1


test_dataset = dataset.H5PY("../training/data/preprocessed/new/BSD/test.h5")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

psnr_val = torch.zeros(len(test_dataloader))
ssim_val = torch.zeros(len(test_dataloader))
n_restart_val = torch.zeros(len(test_dataloader))
n_iter_val = torch.zeros(len(test_dataloader))
indexes = torch.zeros(len(test_dataloader))

sigma = 10

lmbd = 10
mu = 1

for idx, im in enumerate(test_dataloader):
    im = im.to(device)
    im_noisy = im + sigma/255*torch.empty_like(im).normal_()
    im_noisy.requires_grad = False
    im_init = im_noisy
    im_denoised = AdaAGD_Recon(im_noisy, model = model, lmbd = lmbd, mu = mu, H = H, Ht = Ht, op_norm = opnorm) 
    psnr_val[idx] = psnr(im_denoised, im, data_range=1)
    ssim_val[idx] = ssim(im_denoised, im)


fig = plt.figure(figsize=(20,10))
gs = GridSpec(nrows= 1, ncols = 2)

ax0 = fig.add_subplot(gs[0])
ax0.set_title("PSNR")
ax0.plot(psnr_val, label = "PSNR")

ax1 = fig.add_subplot(gs[1])
ax1.set_title("SSIM")
ax1.plot(ssim_val, label = "SSIM")


fileName = "QuantitativCRRNNSigma5.png"
path = os.path.join("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_CRRNN_Sigma_5", fileName) 
plt.savefig(path)
plt.show()