import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr

import sys
sys.path.append('../')
sys.path.append('C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers')
sys.path.append('C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizer/inverse_problems')
# sys.path.append('../models')
# sys.path.append('../inverse_problems')
# sys.path.append('../convex_ridge_regularizers')
import models.utils as utils
import cv2
import math
import matplotlib.pyplot as plt
from inverse_problems.utils_inverse_problems.reconstruction_map_crr import AdaGD_Recon, AGD_Recon

device = 'cpu'

torch.set_grad_enabled(False)

sigma_training = 5
t = 10

exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)

model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)


# intialize the eigen vector of dimension (size, size) associated to the largest eigen value
model.initializeEigen(size=100)
# compute bound via a power iteration which couples the activations and the convolutions
model.precise_lipschitz_bound(n_iter=100)
# the bound is stored in the model
L = model.L.data.item()

img = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/tutorial/image/sample.JPG", cv2.IMREAD_GRAYSCALE), (504, 378))
img_torch = torch.tensor(img, device=device).reshape((1,1) + img.shape)/255
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)


# define the forward operator and its adjoint
# here, for denoising Id
H = lambda x: x
Ht = lambda x: x

lmbd = 25
mu = 4

# x_out, psnr_, ssim_, n_iter = AdaGD_Recon(y=img_torch_noisy, H=H, Ht=Ht, model=model, lmbd=lmbd, mu=mu, x_gt=img_torch, tol=1e-6, max_iter=200)

x_out, psnr_, ssim_, n_iter = AGD_Recon(y=img_torch_noisy, H=H, Ht=Ht, model=model, lmbd=25, mu=3, x_gt=img_torch, tol=1e-5, strong_convexity_constant=1, max_iter=200)


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
ax[0].set_title(f"Clean Image (Reg Cost {model.cost(mu*img_torch)[0].item():.1f})")
ax[0].imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[0].set_yticks([])
ax[0].set_xticks([])

ax[1].set_title(f"Noisy Image (Reg Cost {model.cost(mu*img_torch_noisy)[0].item():.1f})")
ax[1].imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1].set_yticks([])
ax[1].set_xticks([])

ax[2].set_title(f"Denoised Image (Regularization Cost {model.cost(mu*x_out)[0].item():.1f})")
ax[2].imshow(x_out.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[2].set_yticks([])
ax[2].set_xticks([])
plt.show()