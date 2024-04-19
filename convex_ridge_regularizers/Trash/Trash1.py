import numpy as np
import argparse
import json
import torch
import torch.autograd as autograd
from PIL import Image
import math
import cv2
import torch.nn as nn
from functools import partial
import pylops_gpu
from torchmetrics.functional import structural_similarity_index_measure as ssim
import math

import sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsTv import MoreauProximator, LinearOperator, LinearOperatorBatch

a = torch.randn(190, 1, 300, 300)

b = a.view(2,-1)

print(b.shape)

