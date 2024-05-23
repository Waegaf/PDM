import argparse
import json
import torch
import os, sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/weakly_convex_ridge_regularizer")
from Infimal_Conv.InfimalConvolution.Training.TrainerInfConv import TrainerInfConv
import os
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def main(device):
    
    # Set up directories for saving results

    config_path = 'C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/weakly_convex_ridge_regularizer/Infimal_Conv/InfimalConvolution/Training/configs/configTraining.json'
    config = json.load(open(config_path))

    exp_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    seed = 561
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    trainer_inst = TrainerInfConv(config, seed, device)
   
    trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='device to use')

    args = parser.parse_args()


    main(args.device)