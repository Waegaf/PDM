import torch
from torch.utils.data import DataLoader
import os, sys
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/training")
# sys.path.append("/cs/research/vision/home2/wgafaiti/Code/convex_ridge_regularizers/training")
from data import dataset
import json
from torch.utils import tensorboard
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure 
from torchmetrics.functional import peak_signal_noise_ratio as psnr

ssim = StructuralSimilarityIndexMeasure()
sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers")
sys.path.append("/cs/research/vision/home2/wgafaiti/Code/convex_ridge_regularizers")
from models.utils import build_model
from Infimal_Conv.utilsInfimalConvolution.utilsInfConvTraining import tstepInfConvDenoiser

# sys.path.append("C:/Users/waelg/OneDrive/Bureau/EPFL_5_2/Code/convex_ridge_regularizers/Infimal_Conv/utilsInfimalConvolution")
sys.path.append("/cs/research/vision/home2/wgafaiti/Code/convex_ridge_regularizers/Infimal_conv/utilsInfimalConvolution")
from utilsInfConvTraining import CRR_NN_Solver_Training, H_fixedPoint, TV_Solver_Training, fixedPointP, JacobianFixedPointP
from utilsTV import MoreauProximator


class TrainerInfConv:

    def __init__(self, config, seed, device):
        global ssim
        ssim = ssim.to(device)
        self.config = config
        self.seed = seed
        self.device = device
        self.sigma = config['sigma']
        self.alpha = config['alpha']


        # datasets and dataloaders
        train_dataset = dataset.H5PY(config['train_dataloader']['train_data_file'])
        val_dataset = dataset.H5PY(config['val_dataloader']['val_data_file'])
        self.batch_size = config['train_dataloader']['batch_size']

        self.train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = config['train_dataloader']['shuffle'], num_workers = config['train_dataloader']['num_workers'], drop_last = True)
        self.val_dataloader = DataLoader(val_dataset,  batch_size = config['val_dataloader']['batch_size'], shuffle = config['val_dataloader']['shuffle'], num_workers = config['val_dataloader']['num_workers'])

        print("---------------------")
        print(f' - Training set: {len(self.train_dataloader)} batches')
        print(f' - Validation set: {len(self.val_dataloader)} samples')

        # Build the model
        self.model, self.config = build_model(self.config)
        self.model = self.model.to(device)

        print("Number od parameters for training: ", self.model.num_params)
        config["number_of_parameters"] = self.model.num_params

        # Set up the optimizers
        self.set_optimization()
        self.epochs = config['training_options']['epochs']

        self.denoise = tstepInfConvDenoiser


        self.criterion = torch.nn.L1Loss(reduction = 'sum')

        self.save_epoch = config["logging_info"]['save_epoch']


        # CHECKPOINTS & TENSOROARD
        self.checkpoint_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        config_save_path = os.path.join(config['logging_info']['log_dir'], config['exp_name'], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent = 4, sort_keys = True)

        writer_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)


    def set_optimization(self):
        ''' There are different optimizers for different parameters'''
        # Choice of the optimizer
        optimizer = torch.optim.Adam

        self.optimizers = []
        model = self.model

        params_conv = model.conv_layer.parameters()
        lr_conv = self.config['optimizer']['lr_conv']
        optimizer_conv = optimizer(params_conv, lr = lr_conv)
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

        self.model.train()


        tbar = tqdm(self.train_dataloader, ncols = 80, position = 0, leave = True)
        self.epoch = epoch

        log = {}

        
        lmbdLagrange = self.config["training_options"]["Lagrange"]
        alpha = self.config["model"]["alpha"]
        lambdaTV = lmbdLagrange/alpha
        moreauProxBatch = MoreauProximator([40, 40], lambdaTV, [None, None], device = self.device, batch = True)
        moreauProx = MoreauProximator([40,40], lambdaTV, [None, None], device = self.device, batch = False)
        for batch_idx, data in enumerate(tbar):
            data = data.to(self.device)
            noise = self.sigma/255 * torch.randn(data.shape, device = data.device)
            noisy_data = data + noise
            nsamples = data.shape[0]

            for i in range(len(self.optimizers)):
                self.optimizers[i].zero_grad()

            # t-step denoiser
            t_steps=self.config["training_options"]["t_steps"]
            # Initialization steps

            z = torch.zeros_like(noisy_data)
            w = torch.zeros_like(noisy_data)
            g = torch.zeros_like(noisy_data)
            Theta1 = torch.zeros_like(noisy_data)
            Theta2 = torch.zeros_like(noisy_data)

            # Differentiable steps
            JacobiansP_OuterLoop = []
            JacobiansW_OuterLoop = []

            for t in range(t_steps):
                # u - optimization
                u = (1/2*lmbdLagrange+1)*(noisy_data + lmbdLagrange*(z+w+Theta1+g-Theta2))
                # z - optimization
                with torch.no_grad():
                    P = moreauProxBatch.batch_applyProxPrimalDual(u - w - Theta1, alpha = 1.0)
                    flattenOuputP = []
                    for nsample in range(nsamples):
                        flattenOuputP.append(P[nsample,...].view(-1))
                Pref = P[nsample]
                tau = 1/8.
                JacobiansP = []
                samplesZ = []
                for nsample in range(nsamples):
                    flattenSampleP = flattenOuputP[nsample]
                    data = (u[nsample,...] - w[nsample,...] - Theta1[nsample,...]).unsqueeze(0)
                    
                    flattenSampleP = flattenSampleP - fixedPointP(flattenSampleP.view_as(P[0,...]), g = data, lmbd= lambdaTV, tau = tau, batch = False, device = self.device)
                    with torch.no_grad():
                        JacobianP = JacobianFixedPointP( flattenOuputP.view_as(Pref), img = data, sigma = 1/tau , lmbd = lambdaTV, device = self.device)

                    JacobiansP.append(JacobianP) 
                    flattenSampleP.register_hook(lambda grad, ns = nsample, tstep =t: torch.linalg.solve(JacobiansP_OuterLoop[tstep][ns].transpose(0,1), grad))
                    P = flattenSampleP.view_as(Pref)
                    Z = data - lambdaTV*moreauProx.batchL.batch_applyL(P)
                    samplesZ.append(Z)
                
                ZBatches = torch.stack(samplesZ, 0)

                # w - optimization
                with torch.no_grad():
                    noisy_data = u - ZBatches - Theta1
                    wOutput = CRR_NN_Solver_Training(noisy_data, self.model, lmbd = (1/lmbdLagrange) * self.model.lmbd_transformed, mu = self.model.mu_transformed, max_iter = 200, batch = True, enforce_positivity = True, device = self.device)
                    flattenOutputW = []
                    for nsample in range(nsamples):
                        flattenOutputW.append(wOutput[nsample,...].view(-1))
                wOutputRef = wOutput[nsample,...]
                JacobiansW = []
                samplesW = []
                Id =torch.eye(1600, device=self.device)
                # Computation of the jacobian matrix of the implicit function for each sample of the batch
                for nsample in range(nsamples):
                    flattenSampleW = flattenOutputW[nsample]
                    flattenOutputW = flattenOutputW - H_fixedPoint(flattenSampleW.view_as(wOutputRef), self.model, noisy_data[nsample,...], lmbdLagrange, beta = self.model.lmbd_transformed, mu = self.model.mu_transformed).view(-1)
                    with torch.no_grad():
                        JacobianW = self.model.mu_transformed*self.model.lmbd_transformed*self.model.Hessian(self.model.mu_transformed*flattenOutputW.view_as(wOutputRef)).reshape(1600,1600)+Id
                    JacobiansW.append(JacobianW)
                    flattenSampleW.register_hook(lambda grad, ns = nsample, tstep = t: torch.linalg.solve(JacobiansW_OuterLoop[tstep][ns].transpose(0,1), grad))
                    samplesW.append(flattenSampleW.view_as(wOutputRef))
                WBatches = torch.stack(samplesW, 0)

                # g - optimization
                g = torch.clip(u + Theta2, 0)

                # Theta1 - optimization
                Theta1 = Theta1 -u + ZBatches + WBatches
                
                # Theta2 - optimization
                Theta2 = Theta2 + u - g
            JacobiansP_OuterLoop.append(JacobiansP)
            JacobiansW_OuterLoop.append(JacobiansW)    
            # data fidelity normalizedd
            data_fidelity = (self.criterion(u, data)) / (data.shape[0]) * 40 * 30 / data.shape[2] / data.shape[3]

            # Regularization

            if self.config['training_options']['tv2_lambda'] > 0 and self.model.use_splines:
                tv2 = self.model.TV2()
                regularization = self.config['training_options']['tv2_lambda'] *self.sigma * tv2
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

            if batch_idx % 100 == 0:
                self.save_checkpoint(epoch, batch_idx)

            tbar.set_description('T ({}) | loss {:.4f}  lmbd {:.3f} | mu {:.3f} | TV2 {:.3f}'.format(epoch, log['loss'],  log['lmbd'], log['mu'], self.model.TV2()))

    
    def optimizer_step(self):

        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

    def valid_epoch(self, epoch):
        self.model.eval()

        loss_val = 0.0
        psnr_val = 0.0
        ssim_val = 0.0

        tbar_val = tqdm(self.val_dataloader, ncols = 40, position = 0, leave = True)

        for batch_idx, data in enumerate(tbar_val):
            data = data.to(self.device)
            noise = (self.sigma/255.0) * torch.randn(data.shape, device = data.device)
            noisy_data = data + noise

            with torch.no_grad():
                output = self.denoise(self.model, noisy_data, t_steps = self.config["training_options"]["t_steps"], alpha = self.alpha)

                loss = self.criterion(output, data)

                out_val = torch.clamp(output, 0., 0.1)

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

    
    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'config': self.config,
            'u': self.model.u
        }
        for i in range(len(self.optimizers)):
            state['optimizer_' + str(i+1) + '_state_dict'] = self.optimizers[i].state_dict()

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint_' + str(epoch) + '.pth'
        torch.save(state, filename)