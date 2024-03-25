import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
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
img = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Mountain.jpg", cv2.IMREAD_GRAYSCALE), (504, 378))
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255

#Creation of the noisy image by adding some normal noise
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)


# Set of the hyperparameters to tune
lmbd = 5e-1


# Reconstruction of the denoised image (using only the proximal operator)
img_denoised = Tv_denoising_reconstruction(img_torch_noisy, lmbd, x_init = img_torch_noisy)


# Computation of the L2 error matrix
L2error = torch.pow(img_denoised - img_torch, 2)


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 6))


ax[0][0].set_title("Clean Image ")
ax[0][0].imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[0][0].set_yticks([])
ax[0][0].set_xticks([])

ax[0][1].set_title("Noisy Image ")
ax[0][1].imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[0][1].set_yticks([]) 
ax[0][1].set_xticks([])


ax[1][0].set_title("Denoised Image")
ax[1][0].imshow(img_denoised.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][0].set_yticks([])
ax[1][0].set_xticks([])

ax[1][1].set_title("L2 error")
ax[1][1].matshow(L2error.squeeze().detach().numpy())
ax[1][1].set_yticks([])
ax[1][1].set_xticks([])

fileName = f"2503DenoisedMountain{lmbd*100:.0f}.png"
path = os.path.join("Infimal_Conv/tv/ResultsTV", fileName) 
plt.savefig(path)

plt.show()