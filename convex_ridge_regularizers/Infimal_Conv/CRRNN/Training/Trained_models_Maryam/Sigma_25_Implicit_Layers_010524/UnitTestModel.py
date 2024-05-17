import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
import json
from matplotlib.gridspec import GridSpec
# Add of the corresponding paths to be enable to use the wanted functions
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
from models import utils
from reconstruction_map_crr import AdaGD_Recon, AdaAGD_Recon

# Creation of a folder to save the results
if not os.path.exists("Infimal_Conv/CRRNN/ResultsCRRNN"):
    os.makedirs("Infimal_Conv/CRRNN/ResultsCRRNN")

# Set seed for the noise added
torch.manual_seed(61)

# Choice of the (CRRNN) model
device = 'cpu'
# sigma_training = 5
# t = 10
# exp_name = f'Sigma_{sigma_training}_t_{t}'

# Load of the model
config_path = "C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/CRRNN/Training/Trained_models_Maryam/Sigma_25_Implicit_Layers_010524/config.json"
checkpoint_path = "C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/CRRNN/Training/Trained_models_Maryam/Sigma_25_Implicit_Layers_010524/checkpoints/checkpoint_01200.pth"
config = json.load(open(config_path))
model, _ = utils.build_model(config)
checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

model.to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
# We need the identity operator and its adjoint (also identity) since we need to specify them to solve the denoising task with the CRRNN
H = lambda x: x
Ht = lambda x: x
# Upperbound of the Lip. Const. of the gradient of || H(x) - y ||^2, which is 1 in the identity case
opnorm = 1

# Uncomment, this once on the VM
# model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)

# Load of the image (.jpg)
img = cv2.imread("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/Images/Lenna.png", cv2.IMREAD_GRAYSCALE)
img_torch = torch.tensor(img, device = device).reshape((1,1) + img.shape)/255

# Creation of the noisy image by adding some normal noise
img_torch_noisy = img_torch + 25/255 * torch.randn_like(img_torch)

# Set of the hyperparameters to tune
lmbd = 20
mu = 1

# Reconstruction of the denoised image
img_denoised, psnrImg, ssimImg, nIterImg, regCost = AdaAGD_Recon(img_torch_noisy, model = model, lmbd = lmbd, mu = mu, H = H, Ht = Ht, op_norm = opnorm, x_gt = img_torch, track_cost = True) 

# Computation of the L2 error matrix
L2Error = torch.pow(img_denoised - img_torch, 2)

fig = plt.figure(figsize=(20,10))
gs = GridSpec(nrows= 2, ncols = 3)

ax0 = fig.add_subplot(gs[0,0])
ax0.set_title("L2 Error")
caxes  = ax0.matshow(L2Error.squeeze().detach().numpy(), vmin = 0.0, vmax = 0.05)
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
ax3.set_title(f"Clean Image (Reg Cost {model.cost(img_torch)[0].item():.1f})")
ax3.imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax3.set_yticks([])
ax3.set_xticks([])

ax4 = fig.add_subplot(gs[1,1])
ax4.set_title(f"Noisy Image (Reg Cost {model.cost(img_torch_noisy)[0].item():.1f})")
ax4.imshow(img_torch_noisy.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax4.set_yticks([])
ax4.set_xticks([])


ax5 = fig.add_subplot(gs[1,2])
ax5.set_title(f"Denoised Image (Regularization Cost {model.cost(img_denoised)[0].item():.1f})")
ax5.imshow(img_denoised.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax5.set_yticks([])
ax5.set_xticks([])

fileName = f"1405DenoisedLenna{lmbd:.0f}Sigma25ImplicitLayers.png"
path = os.path.join("Infimal_Conv/CRRNN/Training/Trained_models_Maryam/Sigma_25_Implicit_Layers_010524/Results", fileName)
plt.savefig(path)

plt.show()