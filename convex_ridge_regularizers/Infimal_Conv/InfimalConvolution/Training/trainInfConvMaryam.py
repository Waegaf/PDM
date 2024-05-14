import sys, os

import argparse
import json
import torch
from TrainerInfConv import TrainerInfConv


def main(device):


    config_path = "/cs/research/vision/home2/wgafaiti/Code/convex_ridge_regularizers/Infimal_conv/InfimalConvolution/Training/configs/InfConvConfig_Maryam.json"
    config = json.load(open(config_path))

    exp_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    seed = 561
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    trainerAgent = TrainerInfConv(config, seed, device)

    trainerAgent.train()

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description = 'Pytorch Training')
    parser.add_argument('-d', '--device', default = "cuda", type = str, help = 'device to use')

    args = parser.parse_args()

    main(args.device)