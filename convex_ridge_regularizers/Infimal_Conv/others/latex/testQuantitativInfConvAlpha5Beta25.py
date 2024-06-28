import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.gridspec import GridSpec
import pandas as pd
import json
from torch.utils.data import DataLoader
# Add of the corresponding paths to be enable to use the wanted functions
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/models")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/training/data")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
import utils
import dataset
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTvCRRNN import reconstruction_map_InfTVCRRNN



# Creation of a folder to save the results
if not os.path.exists("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_InfConv"):
    os.makedirs("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_InfConv")

# Set seed for the noise added
torch.manual_seed(61)

# Choice of the (CRRNN) model
device = 'cpu'
# Load of the checkpoints path to the model
# config_path = "C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/CRRNN/Training/Trained_models_Maryam/Sigma_5_Implicit_Layers_020524/config.json"
# checkpoint_path = "C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/CRRNN/Training/Trained_models_Maryam/Sigma_5_Implicit_Layers_020524/checkpoints/checkpoint_01200.pth"
# config = json.load(open(config_path))
# model, _ = utils.build_model(config)
# checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

# model.to(device)
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()

# Load model
config_path = "C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/InfimalConvolution/Training/Trained_models/InfConv_Sigma_5_t_1_2605/config.json"
checkpoint_path = "C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/InfimalConvolution/Training/Trained_models/InfConv_Sigma_5_t_1_2605/checkpoints/checkpoint_61020.pth"
config = json.load(open(config_path))
model, _ = utils.build_model(config)
checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

model.to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# # Load model
# sigma_training = 25
# t = 10
# exp_name = f'Sigma_{sigma_training}_t_{t}'
# model = utils.load_model(exp_name, device)


# We need the identity operator and its adjoint (also identity) since we need to specify them to solve the denoising task with the CRRNN
H = lambda x: x
Ht = lambda x: x
# Upperbound of the Lip. Const. of the gradient of || H(x) - y ||^2, which is 1 in the identity case
opnorm = 1


test_dataset = dataset.H5PY("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/preprocessed/new/BSD/test.h5")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

psnr_val = torch.zeros(len(test_dataloader))
ssim_val = torch.zeros(len(test_dataloader))
n_restart_val = torch.zeros(len(test_dataloader))
n_iter_val = torch.zeros(len(test_dataloader))
indexes = torch.zeros(len(test_dataloader))

sigma = 10

lmbdLagrange = 1e-1
alpha = 5e-2
beta = 7

with torch.no_grad():
    for idx, im in enumerate(test_dataloader):
        im = im.to(device)
        im_noisy = im + sigma/255*torch.empty_like(im).normal_()
        im_noisy.requires_grad = False
        im_init = im_noisy
        im_denoised, _, _, _, _, _ = reconstruction_map_InfTVCRRNN(model = model, x_noisy=im_noisy, lmbdLagrange=lmbdLagrange, alpha = alpha, beta = beta, maxIter= 300)
        psnr_val[idx] = psnr(im_denoised, im, data_range=1)
        ssim_val[idx] = ssim(im_denoised, im)

psnr_ = psnr_val.mean().item()
ssim_ = ssim_val.mean().item()

path = f"C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_InfConv/Quantitativ_Trained_Sigma_25_{beta}_csv.csv"
columns = ["sigma_test",  "model_name", "psnr", "ssim"]
if os.path.isfile(path):
    db = pd.read_csv(path)
else:
    db = pd.DataFrame(columns=columns)

line = {"sigma_test": sigma,  "model_name": "InfConv"}

ind = [True] * len(db)
for col, val in line.items():
    ind = ind & (db[col] == val)
db = db.drop(db[ind].index)
line["psnr"] = psnr_
line["ssim"] = ssim_
db = pd.concat((db, pd.DataFrame([line], columns=columns)), ignore_index=True)

db.to_csv(path, index=False)