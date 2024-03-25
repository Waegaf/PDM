import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
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
img = cv2.resize(cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/GrassHybridCircle.jpg", cv2.IMREAD_GRAYSCALE), (504, 378))
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255

# Creation of the noisy image by adding some normal noise
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)

# Set of the hyperparameters to tune (see Master Thesis for more details)
lmbdLagrange = 1
alpha = 5e-2 
beta = 25

# image_cleaned corresponds to u in the computations
img_denoised, z, w, g, psnrImg, ssimImg, regCost = reconstruction_map_InfTVCRRNN(model = model, x_noisy=img_torch_noisy, lmbdLagrange=lmbdLagrange, alpha = alpha, beta = beta, maxIter= 200, x_origin=img_torch, maxIterTVRecon = 100, maxIterCRRNNRecon = 100, trackCost = True)

# Computation of the L2 error matrix
L2Error = torch.pow(img_denoised - img_torch, 2)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))

ax[0][1].set_title("Track of the peak signal noise ratio")
ax[0][1].plot(psnrImg, label = 'psnr')

ax[0][2].set_title('Track of the structural similarity index measure')
ax[0][2].plot(ssimImg, label = 'ssim')


ax[1][0].set_title(f"Clean Image (Reg Cost {Regularization_cost_InfTVCRRNN(img_torch, img_torch, img_torch, img_torch, model, lmbdLagrange, alpha, beta):.1f})")
ax[1][0].imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][0].set_yticks([])
ax[1][0].set_xticks([])

ax[1][1].set_title("L2 Error")
ax[1][1].matshow(L2Error.squeeze().detach().numpy())
ax[1][1].set_title(f"Noisy Image (Reg Cost {Regularization_cost_InfTVCRRNN(img_torch_noisy, img_torch_noisy, img_torch_noisy, img_torch_noisy, model, lmbdLagrange, alpha, beta):.1f})")
ax[1][1].imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][1].set_yticks([])
ax[1][1].set_xticks([])


ax[1][2].set_title(f"Denoised Image (Regularization Cost {regCost[-1]:.1f})")
ax[1][2].imshow(img_denoised.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1][2].set_yticks([])
ax[1][2].set_xticks([])

ax[0][0].set_title("L2 Error")
ax[0][0].matshow(L2Error.squeeze().detach().numpy())
ax[0][0].set_yticks([])
ax[0][0].set_xticks([])

fileName = f"2503DenoisedGrassHybridCircleA{alpha*100:.0f}B{beta:.0f}S{sigma_training:.0f}t{t:.0f}.png"
path = os.path.join("Infimal_Conv/InfimalConvolution/ResultsInfConv", fileName) 
plt.savefig(path)

plt.show()