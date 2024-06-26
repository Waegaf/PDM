import sys, os

import argparse
import json
import torch
from TrainerCRRNN import TrainerCRRNN


def main(device):

    config_path = 'C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/CRRNN/Training/configs/CRRNNTestConfig.json'
    config = json.load(open(config_path))

    exp_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    seed = 561
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    trainerAgent = TrainerCRRNN(config, seed, device)

    trainerAgent.train()

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description = 'Pytorch Training')
    parser.add_argument('-d', '--device', default = "cpu", type = str, help = 'device to use')

    args = parser.parse_args()

    main(args.device)