import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
from models.utilsTvCRRNN import reconstruction_map_InfTVCRRNN, Regularization_cost_InfTVCRRNN
from models import utils


# Choice of the (CRRNN) model
device = 'cpu'
sigma_training = 5
t = 10
exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)
# Uncomment, this once on the VM
# model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)


# Load of the image (.jpg)
img = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Chasseron.jpg", cv2.IMREAD_GRAYSCALE), (504, 378))
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)


# Set of the hyperparameters
lmbdLagrange = 1
alpha = 5  
beta = 10

# image_cleaned corresponds to u in the context of minimization
img_cleaned, z, w, g, psnrImg, ssimImg, regCost = reconstruction_map_InfTVCRRNN(model = model, x_noisy=img_torch_noisy, lmbdLagrange=lmbdLagrange, alpha = alpha, beta = beta, maxIter= 200, x_origin=img_torch, maxIterTVRecon = 40, maxIterCRRNNRecon = 40, trackCost = True)


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
ax[0][0].set_title("Track of the regularization cost")
ax[0][0].plot(regCost, label = 'reg. cost')

ax[0][1].set_title("Track of the peak signal noise ratio")
ax[0][1].plot(psnrImg, label = 'psnr')

ax[0][2].set_title('Track of the structural similarity index measure')
ax[0][2].plot(ssimImg, label = 'ssim')


ax[1][0].set_title(f"Clean Image (Reg Cost {Regularization_cost_InfTVCRRNN(img_torch, img_torch, img_torch, img_torch, model, lmbdLagrange, alpha, beta):.1f})")
ax[1][0].imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][0].set_yticks([])
ax[1][0].set_xticks([])

ax[1][1].set_title(f"Noisy Image (Reg Cost {Regularization_cost_InfTVCRRNN(img_torch_noisy, img_torch_noisy, img_torch_noisy, img_torch_noisy, model, lmbdLagrange, alpha, beta):.1f})")
ax[1][1].imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][1].set_yticks([])
ax[1][1].set_xticks([])

ax[1][2].set_title(f"Denoised Image (Regularization Cost {regCost[-1]:.1f})")
ax[1][2].imshow(img_cleaned.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][2].set_yticks([])
ax[1][2].set_xticks([])

plt.show()