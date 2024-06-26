import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/training")
from data import dataset
import os
import json
from torch.utils import tensorboard
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import torch.autograd as autograd
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
from models.utils import build_model
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
from utilsInfConvTraining import CRR_NN_Solver_Training, H_fixedPoint

ssim = StructuralSimilarityIndexMeasure()

# Construction of the Trainer class using IMPLICIT LAYERS approach


class TrainerCRRNN:
    """
    """

    def __init__(self, config, seed, device):
        global ssim
        ssim = ssim.to(device)
        self.config = config
        self.seed = seed
        self.device = device
        self.sigma = config['sigma']

        # datasets and dataloaders
        train_dataset = dataset.H5PY(config['train_dataloader']['train_data_file'])
        val_dataset = dataset.H5PY(config['val_dataloader']['val_data_file'])
        self.batch_size = config["train_dataloader"]["batch_size"]

        self.train_dataloader = DataLoader(train_dataset, batch_size=config["train_dataloader"]["batch_size"], shuffle=config["train_dataloader"]["shuffle"], num_workers=config["train_dataloader"]["num_workers"], drop_last=True)
        
        self.val_dataloader = DataLoader(val_dataset, batch_size=config["val_dataloader"]["batch_size"], shuffle=config["val_dataloader"]["shuffle"], num_workers=config["val_dataloader"]["num_workers"])

        print("---------------------")
        print(f' - Training set: {len(self.train_dataloader)} batches')
        print(f' - Validation set: {len(self.val_dataloader)} samples') 

        # Build the model
        self.model, self.config = build_model(self.config)
        self.model = self.model.to(device)

        print("Number of parameters for training: ", self.model.num_params)
        config["number_of_parameters"] = self.model.num_params

        # Set up the optimizer
        self.set_optimization()
        self.epochs = config["training_options"]['epochs']

        self.criterion = torch.nn.L1Loss(reduction='sum')
        
        self.save_epoch = config["logging_info"]['save_epoch']

        # CHECKPOINTS & TENSOBOARD
        self.checkpoint_dir = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        config_save_path = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

    def set_optimization(self):

        """ The optimizer are different for different parameters"""
        # optimizer
        optimizer = torch.optim.Adam

        self.optimizers = []
        model = self.model
    
        params_conv = model.conv_layer.parameters()
        lr_conv = self.config["optimizer"]["lr_conv"]
        optimizer_conv = optimizer(params_conv, lr=lr_conv)
        self.optimizers.append(optimizer_conv)

        # activation parameters
        if self.model.use_splines:
            params_activations = [model.activation.coefficients_vect]
            lr_activations = self.config["optimizer"]["lr_spline_coefficients"]
            optimizer_activations = optimizer(params_activations, lr=lr_activations)
            self.optimizers.append(optimizer_activations)
        else:
            params_bias = [model.bias]
            lr_bias= self.config["optimizer"]["lr_spline_coefficients"]
            optimizer_bias = optimizer(params_bias, lr=lr_bias)
            self.optimizers.append(optimizer_bias)  

        # reg strenght learnt
        lr_lmbd = self.config["optimizer"]["lr_lmbd"]
        optimizer_lmbd = optimizer([self.model.lmbd], lr=lr_lmbd)
        self.optimizers.append(optimizer_lmbd)

        # scaling factor learnt
        lr_mu = self.config["optimizer"]["lr_mu"]
        optimizer_mu = optimizer([self.model.mu], lr=lr_mu)
        self.optimizers.append(optimizer_mu)
  
        # scheduler
        if self.config["training_options"]["lr_scheduler"]["use"]:
            self.scheduler = [torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=self.config["training_options"]["lr_scheduler"]["gamma"], verbose=False) for optim in self.optimizers]
    
    def train(self):
        for epoch in range(self.epochs+1):
            self.train_epoch(epoch)
            # validation - if any
            if (epoch % self.config["logging_info"]["epochs_per_val"] == 0):
                self.valid_epoch(epoch)

            # scheduler
            if self.config["training_options"]["lr_scheduler"]["use"] and (epoch < self.config["training_options"]["lr_scheduler"]["nb_steps"]):
                for sc in self.scheduler:
                    sc.step()

            # SAVE CHECKPOINT
            if (epoch % self.save_epoch == 0 and epoch != 0) or (epoch == 0 and self.save_epoch == 1):
                self.save_checkpoint(epoch)
        
        self.writer.flush()
        self.writer.close()


    def train_epoch(self, epoch):
        """
        """

        self.model.train()

        tbar = tqdm(self.train_dataloader, ncols= 80, position = 0, leave = True)
        self.epoch = epoch

        log = {}
        for batch_idx, data in enumerate(tbar):
            data = data.to(self.device)
            noise = self.sigma/255 * torch.randn(data.shape, device = data.device)
            noisy_data = data + noise

            for i in range(len(self.optimizers)):
                self.optimizers[i].zero_grad()

            # Implicit layer
            nbatches = data.shape[0]
            # Solving of the inverse problem (batch) under torch.no_grad()
            with torch.no_grad():
                output = CRR_NN_Solver_Training(noisy_data, self.model, lmbd = self.model.lmbd_transformed, mu = self.model.mu_transformed, max_iter = 200, batch = True, enforce_positivity = True, device = self.device)
                flatten_output = []
                for nsample in range(nbatches):
                    flatten_output.append(output[nsample,...].view(-1))
                output1 = output[nsample,...]
                del output
            # Initialisation of empty lists Jacobians and sample
            Jacobians = []
            samples = []
            Id =torch.eye(1600, device=self.device)
            # Computation of the jacobian matrix of the implicit function for each sample of the batch
            for nsample in range(nbatches):
                flatten_outputSample = flatten_output[nsample]
                flatten_outputSample = flatten_outputSample - H_fixedPoint(flatten_outputSample.view_as(output1), self.model, noisy_data[nsample,...], lmbdLagrange = 1., beta = self.model.lmbd_transformed, mu = self.model.mu_transformed).view(-1)
                # Computation of the jacobian ("manually") for each sample of the batch
                with torch.no_grad():
                    JacobianManual = self.model.mu_transformed*self.model.lmbd_transformed*self.model.Hessian(self.model.mu_transformed*flatten_outputSample.view_as(output1)).reshape(1600,1600)+Id
                Jacobians.append(JacobianManual)
                flatten_outputSample.register_hook(lambda grad, ns = nsample: torch.linalg.solve(Jacobians[ns].transpose(0,1), grad))
                samples.append(flatten_outputSample.view_as(output1))
            finalOutput = torch.stack(samples, 0)
            # data fidelity normalized
            data_fidelity = (self.criterion(finalOutput, data))/(data.shape[0]) * 40 * 40 / data.shape[2] / data.shape[3]

            # regularization of the splines to promote fewer breakpoints
            if self.config['training_options']['tv2_lambda'] > 0 and self.model.use_splines:
                tv2 = self.model.TV2()
                regularization = self.config['training_options']['tv2_lambda'] * self.sigma * tv2
            else:
                regularization = 0

            total_loss = data_fidelity + regularization
            total_loss.backward()

            log['loss'] = data_fidelity
            log['lmbd'] = (self.model.lmbd_transformed).item()
            log['mu'] = (self.model.mu_transformed).item()
            if self.config['training_options']['tv2_lambda'] > 0 and self.model.use_splines:
                log['tv2'] = tv2.item()

            self.optimizer_step()
            self.wrt_step = (epoch) * len(self.train_dataloader) + batch_idx
            self.write_scalars_tb(log)

            # Save checkpoint every 100 batces
            if batch_idx % 100 == 0:
                self.save_checkpoint(epoch, batch_idx)

            tbar.set_description('T ({}) | loss {:.4f}  | lmbd {:.3f} | mu {:.3f} | TV2 {:.3f}'.format(epoch, log['loss'], log['lmbd'], log['mu'], self.model.TV2()))

    def optimizer_step(self):
        """ """
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()


    def valid_epoch(self, epoch):
        self.model.eval()

        loss_val = 0.0
        psnr_val = 0.0
        ssim_val = 0.0

        tbar_val = tqdm(self.val_dataloader, ncols=40, position=0, leave=True)
        
        for batch_idx, data in enumerate(tbar_val):
            data = data.to(self.device)
            noise = (self.sigma/255.0)*torch.randn(data.shape,device=data.device)
            noisy_data = data + noise

            with torch.no_grad():
                output = CRR_NN_Solver_Training(noisy_data, self.model, lmbd = self.model.lmbd_transformed, mu = self.model.mu_transformed, max_iter = 200, batch = True, enforce_positivity = True, device = self.device)

                loss = self.criterion(output, data)
                out_val = torch.clamp(output, 0., 1.)

                loss_val = loss_val + loss.cpu().item()
                psnr_val = psnr_val + psnr(out_val, data, 1.).mean().item()
                ssim_val = ssim_val + ssim(out_val, data).mean().item()
                
        # PRINT INFO
        loss_val = loss_val/len(self.val_dataloader)
        tbar_val.set_description('  Validation ({}) - PSNR:{:.3f}'.format(epoch, psnr_val))

        # METRICS TO TENSORBOARD
        self.wrt_mode = 'Convolutional'
        self.writer.add_scalar(f'{self.wrt_mode}/Validation loss', loss_val, epoch)
        psnr_val = psnr_val/len(self.val_dataloader)
        ssim_val = ssim_val/len(self.val_dataloader)
        self.writer.add_scalar(f'{self.wrt_mode}/Validation PSNR', psnr_val, epoch)
        self.writer.add_scalar(f'{self.wrt_mode}/Validation SSIM', ssim_val, epoch)

        print('EVAL ({}) | MSELoss: {:.5f} |PSNR:{:.4f}'.format(epoch, loss_val,psnr_val))


        log = {'val_loss': loss_val}
        log["val_psnr"] = psnr_val
        log["val_ssim"] = ssim_val



    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'Convolutional/Training {k}', v, self.wrt_step)

    def save_checkpoint(self, epoch, batch_idx = 9):
        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'state_dict': self.model.state_dict(),
            'config': self.config,
            'u': self.model.u
        }
        for i in range(len(self.optimizers)):
            state['optimizer_' + str(i+1) + '_state_dict'] = self.optimizers[i].state_dict()

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint_' + str(epoch) + str(batch_idx) +'.pth'
        torch.save(state, filename)