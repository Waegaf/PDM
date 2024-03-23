import numpy as np
import argparse
import json
import torch
from PIL import Image
import math


a = torch.randn([1,1,20,20])
print(a)

b = torch.clamp(a, None, None)

print(b)


