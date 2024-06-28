import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.gridspec import GridSpec
import json
import pandas as pd
from torch.utils.data import DataLoader
# Add of the corresponding paths to be enable to use the wanted functions
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/inverse_problems/utils_inverse_problems")
from models import utils
from training.data import dataset
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTv import TV_reconstruction, Tv_denoising_reconstruction



# Creation of a folder to save the results
if not os.path.exists("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_TV"):
    os.makedirs("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_TV")

# Set seed for the noise added
torch.manual_seed(61)

# Choice of the (CRRNN) model
device = 'cpu'

test_dataset = dataset.H5PY("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/preprocessed/new/BSD/test.h5")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

psnr_val = torch.zeros(len(test_dataloader))
ssim_val = torch.zeros(len(test_dataloader))
n_restart_val = torch.zeros(len(test_dataloader))
n_iter_val = torch.zeros(len(test_dataloader))
indexes = torch.zeros(len(test_dataloader))

sigma = 10

lmbd = 5e-2

for idx, im in enumerate(test_dataloader):
    im = im.to(device)
    im_noisy = im + sigma/255*torch.empty_like(im).normal_()
    im_noisy.requires_grad = False
    im_init = im_noisy
    im_denoised = Tv_denoising_reconstruction(im_noisy, lmbd)
    psnr_val[idx] = psnr(im_denoised, im, data_range=1)
    ssim_val[idx] = ssim(im_denoised, im)

psnr_ = psnr_val.mean().item()
ssim_ = ssim_val.mean().item()

path = "C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/others/latex/Results_Quant_TV/Quantitativ_csv.csv"
columns = ["sigma_test",  "model_name", "psnr", "ssim"]
if os.path.isfile(path):
    db = pd.read_csv(path)
else:
    db = pd.DataFrame(columns=columns)

line = {"sigma_test": sigma,  "model_name": "TV"}

ind = [True] * len(db)
for col, val in line.items():
    ind = ind & (db[col] == val)
db = db.drop(db[ind].index)
line["psnr"] = psnr_
line["ssim"] = ssim_
db = pd.concat((db, pd.DataFrame([line], columns=columns)), ignore_index=True)

db.to_csv(path, index=False)


