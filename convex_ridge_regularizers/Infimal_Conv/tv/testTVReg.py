import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
from models import utils
from reconstruction_map_tv import TV_Recon
if not os.path.exists("Infimal_Conv/tv/ResultsTV"):
    os.makedirs("Infimal_Conv/tv/ResultsTV")


torch.manual_seed(61)
# Choice of the device
device = 'cpu'


# Load of the image (.jpg)
img = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Chasseron.jpg", cv2.IMREAD_GRAYSCALE), (504, 378))
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)


# Set of the hyperparameters to tune
lmbd = 0.07

# Set of the arguments to perform a TV reconstruction
H = lambda x: x
Ht = lambda x: x
alpha = 1



# image_cleaned corresponds to u in the context of minimization
img_cleaned, psnrImg, ssimImg, nIterImg = TV_Recon(img_torch_noisy, alpha = alpha, lmbd = lmbd, H = H, Ht = Ht, x_gt = img_torch, x_init = img_torch_noisy)
print(f"Results in {nIterImg:.0f} iterations \n ")
print(f"PSNR: at first: {psnrImg[0].item():.1f}, at the end: {psnrImg[-1].item():.1f}")
print(f"SSIM: at first. {ssimImg[0].item():.1f}, at the end: {ssimImg[-1].item():.1f}")

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))


ax[0].set_title("Clean Image ")
ax[0].imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[0].set_yticks([])
ax[0].set_xticks([])

ax[1].set_title("Noisy Image ")
ax[1].imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1].set_yticks([])
ax[1].set_xticks([])


ax[2].set_title("Denoised Image")
ax[2].imshow(img_cleaned.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[2].set_yticks([])
ax[2].set_xticks([])

fileName = f"denoised{lmbd*100:.0f}.png"
path = os.path.join("Infimal_Conv/tv/ResultsTV", fileName) 
plt.savefig(path)

plt.show()