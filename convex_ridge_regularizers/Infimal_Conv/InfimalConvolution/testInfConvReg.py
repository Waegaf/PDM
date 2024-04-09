import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.gridspec import GridSpec
# Add of the corresponding paths to be enable to use the wanted functions
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTvCRRNN import reconstruction_map_InfTVCRRNN, Regularization_cost_InfTVCRRNN
from models import utils

# Creation of a folder to save the results
if not os.path.exists("Infimal_Conv/InfimalConvolution/ResultsInfConv"):
    os.makedirs("Infimal_Conv/InfimalConvolution/ResultsInfConv")


# Set seed for the noise added
torch.manual_seed(61)

# Choice of the (CRRNN) model
device = 'cpu'
sigma_training = 5
t = 10
exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)

# Uncomment, this once on the VM
# model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)

# Load of the image (.jpg)
img = cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Lenna.png", cv2.IMREAD_GRAYSCALE)
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255

# Creation of the noisy image by adding some normal noise
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)

# Set of the hyperparameters to tune (see Master Thesis for more details)
lmbdLagrange = 0.5
alpha = 5e-2 
beta = 30

# image_cleaned corresponds to u in the computations
img_denoised, z, w, g, psnrImg, ssimImg, regCost = reconstruction_map_InfTVCRRNN(model = model, x_noisy=img_torch_noisy, lmbdLagrange=lmbdLagrange, alpha = alpha, beta = beta, maxIter= 200, x_origin=img_torch, maxIterTVRecon = 100, maxIterCRRNNRecon = 100, trackCost = True)

# Computation of the L2 error matrix
L2Error = torch.pow(img_denoised - img_torch, 2)


fig = plt.figure(figsize=(20,10))
gs = GridSpec(nrows= 2, ncols = 3)

ax0 = fig.add_subplot(gs[0,0])
ax0.set_title("L2 Error")
caxes  = ax0.matshow(L2Error.squeeze().detach().numpy(), vmin = 0.0, vmax = 0.02)
fig.colorbar(caxes, ax = ax0)
ax0.set_xticks([])
ax0.set_yticks([])

ax1 = fig.add_subplot(gs[0,1])
ax1.set_title("Track of the peak signal noise ratio")
ax1.plot(psnrImg, label = 'psnr')

ax2 = fig.add_subplot(gs[0,2])
ax2.set_title('Track of the structural similarity index measure')
ax2.plot(ssimImg, label = 'ssim')

ax3 = fig.add_subplot(gs[1,0])
ax3.set_title(f"Clean Image ")
ax3.imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax3.set_yticks([])
ax3.set_xticks([])

ax4 = fig.add_subplot(gs[1,1])
ax4.set_title(f"Noisy Image")
ax4.imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax4.set_yticks([])
ax4.set_xticks([])


ax5 = fig.add_subplot(gs[1,2])
ax5.set_title(f"Denoised Image ")
ax5.imshow(img_denoised.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax5.set_yticks([])
ax5.set_xticks([])

fileName = f"0404LennaA{alpha*100:.0f}B{beta:.0f}S{sigma_training:.0f}t{t:.0f}.png"
path = os.path.join("Infimal_Conv/InfimalConvolution/ResultsInfConv", fileName) 
plt.savefig(path)

plt.show()