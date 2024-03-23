import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
from models import utils
from reconstruction_map_crr import AdaGD_Recon, AdaAGD_Recon
if not os.path.exists("Infimal_Conv/CRRNN/ResultsCRRNN"):
    os.makedirs("Infimal_Conv/CRRNN/ResultsCRRNN")


torch.manual_seed(61)

# Choice of the (CRRNN) model
device = 'cpu'
sigma_training = 5
t = 10
exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)
H = lambda x: x
Ht = lambda x: x
opnorm = 1

# Uncomment, this once on the VM
# model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)


# Load of the image (.jpg)
img = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Chasseron.jpg", cv2.IMREAD_GRAYSCALE), (504, 378))
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)


# Set of the hyperparameters to tune
lmbd = 25

# image_cleaned corresponds to u in the context of minimization
img_cleaned, psnrImg, ssimImg, nIterImg, regCost = AdaAGD_Recon(img_torch_noisy, model = model, lmbd = lmbd, mu = 1, H = H, Ht = Ht, op_norm = opnorm, x_gt = img_torch, track_cost = True, x_init = img_torch_noisy) 


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
ax[0][0].set_title("Track of the regularization cost")
ax[0][0].plot(regCost, label = 'reg. cost')

ax[0][1].set_title("Track of the peak signal noise ratio")
ax[0][1].plot(psnrImg, label = 'psnr')

ax[0][2].set_title('Track of the structural similarity index measure')
ax[0][2].plot(ssimImg, label = 'ssim')


ax[1][0].set_title(f"Clean Image (Reg Cost {model.cost(img_torch)[0].item():.1f})")
ax[1][0].imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][0].set_yticks([])
ax[1][0].set_xticks([])

ax[1][1].set_title(f"Noisy Image (Reg Cost {model.cost(img_torch_noisy)[0].item():.1f})")
ax[1][1].imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][1].set_yticks([])
ax[1][1].set_xticks([])


ax[1][2].set_title(f"Denoised Image (Regularization Cost {model.cost(img_cleaned)[0].item():.1f})")
ax[1][2].imshow(img_cleaned.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][2].set_yticks([])
ax[1][2].set_xticks([])

fileName = f"denoised{lmbd:.0f}Sigma{sigma_training:.0f}tStep{t:.0f}.png"
path = os.path.join("Infimal_Conv/CRRNN/ResultsCRRNN", fileName) 
plt.savefig(path)

plt.show()