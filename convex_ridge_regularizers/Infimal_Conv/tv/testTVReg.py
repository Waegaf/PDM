import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.gridspec import GridSpec
# Add of the corresponding paths to be enable to use the wanted functions
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTv import TV_reconstruction, Tv_denoising_reconstruction

# Creation of a folder to save the results
if not os.path.exists("Infimal_Conv/tv/ResultsTV"):
    os.makedirs("Infimal_Conv/tv/ResultsTV")

# Set seed for the noise added
torch.manual_seed(61)


# Choice of the device (since we can work on the gpu also)
device = 'cpu'

# Load of the image (.jpg)
# img = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Lenna.png", cv2.IMREAD_GRAYSCALE), (504, 378))
img = cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/whiteBackground.png", cv2.IMREAD_GRAYSCALE)
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255

#Creation of the noisy image by adding some normal noise
img_torch_noisy = img_torch + 100/255 * torch.randn_like(img_torch)


# Set of the hyperparameters to tune
lmbd = 5


# Reconstruction of the denoised image (using only the proximal operator)
img_denoised = Tv_denoising_reconstruction(img_torch_noisy, lmbd)


# Computation of the L2 error matrix
L2error = torch.pow(img_denoised - img_torch, 2)


fig = plt.figure()
gs = GridSpec(nrows= 2, ncols = 2)

ax0 = fig.add_subplot(gs[0,0])
ax0.set_title("Clean Image")
ax0.imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax0.set_yticks([])
ax0.set_xticks([])

ax1 = fig.add_subplot(gs[0,1])
ax1.set_title("Noisy Image ")
ax1.imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax1.set_yticks([]) 
ax1.set_xticks([])

ax2 = fig.add_subplot(gs[1,0])
ax2.set_title("Denoised Image")
ax2.imshow(img_denoised.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax2.set_yticks([])
ax2.set_xticks([])

ax3 = fig.add_axes([0.57 , 0.1, 0.35, 0.35])
ax3.set_title("L2 error")
caxes = ax3.matshow(L2error.squeeze().detach().numpy(), vmin = 0.0, vmax = 0.05)
fig.colorbar(caxes, ax = ax3)
ax3.set_yticks([])
ax3.set_xticks([])


fileName = f"2305DenoisedWhiteBackground{lmbd*100:.0f}.png"
path = os.path.join("Infimal_Conv/tv/ResultsTV", fileName) 
plt.savefig(path)

plt.show()