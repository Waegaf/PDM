import argparse
import json
import torch
import trainer
import os

def main():

    device = 'cpu'

    config_path = 'configs/newConfig.json'
    config = json.load(open(config_path))

    exp_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    seed = 61
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    trainer_inst = trainer.Trainer(config, seed, device)
   
    trainer_inst.train()

if __name__ == "__main__":
    main()


