import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
from models.utilsTvCRRNN import reconstruction_map_InfTVCRRNN, Regularization_cost_InfTVCRRNN
from models import utils

device = 'cpu'
sigma_training = 5
t = 50

exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)
# model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)

img = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Chasseron.jpg", cv2.IMREAD_GRAYSCALE), (504, 378))
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)

lmbdLagrange = 1e-1

img_cleaned, z, w, g, psnrImg, ssimImg, regCost = reconstruction_map_InfTVCRRNN(model = model, x_noisy=img_torch_noisy, lmbdLagrange=lmbdLagrange, alpha = 5e-2, beta = 5e-2, maxIter= 10, maxIterTVRecon = 10, maxIterCRRNNRecon = 10)



fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
ax[0].set_title(f"Clean Image (Reg Cost {Regularization_cost_InfTVCRRNN(img_torch, img_torch, img_torch, img_torch, model, lmbdLagrange, 5e-2, 5e-2):.1f})")
ax[0].imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[0].set_yticks([])
ax[0].set_xticks([])

ax[1].set_title(f"Noisy Image (Reg Cost {Regularization_cost_InfTVCRRNN(img_torch_noisy, img_torch_noisy, img_torch_noisy, img_torch_noisy, model, lmbdLagrange, 5e-2, 5e-2):.1f})")
ax[1].imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1].set_yticks([])
ax[1].set_xticks([])

ax[2].set_title(f"Denoised Image (Regularization Cost {regCost:.1f})")
ax[2].imshow(img_cleaned.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[2].set_yticks([])
ax[2].set_xticks([])
plt.show()