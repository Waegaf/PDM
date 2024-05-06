import math
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from models.multi_conv import MultiConv2d
from models.linear_spline import LinearSpline


class ConvexRidgeRegularizer(nn.Module):
    """Module to parametrize a CRR-NN model, mainly gradient-focused"""
    def __init__(self, channels=[1, 8, 32], kernel_size=3, activation_params={"knots_range":0.1, "n_knots": 21}):
        """
        Parameters
        ----------
        channels : list of int
            The list of channels from the input to output
            e.g. [1, 8, 32] for 2 convolutional layers with 1 input channels, 32 output channels and 8 channels in between
        kernel_size : int
            The size of the kernel of all convolutional layers
        activation_params : dict
            The parameters of the activation function"""
        super().__init__()

        padding = kernel_size // 2
        self.padding = padding
        self.channels = channels

        # learnable regularization strength
        self.lmbd = nn.parameter.Parameter(data=torch.tensor(5.), requires_grad=True)
        # learnable regularization scaling
        self.mu = nn.parameter.Parameter(data=torch.tensor(1.), requires_grad=True)

        # linear layer, made of compositions of convolutions
        self.conv_layer = MultiConv2d(channels=channels, kernel_size=kernel_size, padding=padding)

        # activation functions
        self.activation_params = activation_params

        activation_params["n_channels"] = channels[-1]


        if "name" not in activation_params:
            activation_params["name"] = "spline"

        if activation_params["name"]== "ReLU":
            self.activation = nn.ReLU()
            self.bias = nn.parameter.Parameter(data=torch.zeros((1, channels[-1], 1, 1)), requires_grad=True)
            self.use_splines = False
            self.lmbd.data *= 1e-3
        else:
            self.activation = LinearSpline(mode="conv", num_activations=activation_params["n_channels"],
                                size=activation_params["n_knots"],
                                range_=activation_params["knots_range"])
            self.use_splines = True
                                    
        self.num_params = sum(p.numel() for p in self.parameters())

        # initialize random image for caching an estimate of the largest eigen vector for Lipschitz bound computation
        # the size matters little compare to number of iterations, so small patches makes training more efficient
        # + a more precise computation is done at test time
        self.initializeEigen(size=20)
        
        # running estimate of Lipschitz
        self.L = nn.parameter.Parameter(data=torch.tensor(1.), requires_grad=False)

        self.W_Conv = None

        print("---------------------")
        print(f"Building a CRR-NN model with \n - {channels} channels \n splines parameters:")
        print(f"  ({self.activation})")
        print("---------------------")


    def initializeEigen(self, size=100):
        self.u = torch.empty((1,1,size, size)).uniform_()

    @property
    def lmbd_transformed(self):
        # ensure lmbd is nonzero positive
        return(torch.clip(self.lmbd, 0.0001, None))

    @property
    def mu_transformed(self):
        # ensure mu is nonzero positive
        return(torch.clip(self.mu, 0.01, None))


    def forward(self, x):
        # linear layer (a multi-convolution)
        y = self.conv_layer(x)
        # activation
        if not self.use_splines:
            y = y + self.bias
        y = self.activation(y)
        # transposed linear layer
        y = self.conv_layer.transpose(y)
        return(y)

    def grad(self, x):
        return(self.forward(x))

    def update_integrated_params(self):
        for ac in self.activation:
            ac.update_integrated_coeff()

    def cost(self, x):
        s = x.shape
        # first multi convolution layer
        y = self.conv_layer(x)
        # activation
        y = self.activation.integrate(y)

        return(torch.sum(y, dim=tuple(range(1, len(s)))))

    # regularization
    def TV2(self, include_weights=False):
        if self.use_splines:
            return(self.activation.TV2(include_weights=include_weights))
        else:
            return(0)

    
    def precise_lipschitz_bound(self, n_iter=50, differentiable=False):
        with torch.no_grad():
            # vector with the max slope of each activation
            if self.use_splines:
                slope_max = self.activation.slope_max
                if slope_max.max().item() == 0:
                    return(torch.tensor([0.], device = slope_max.device))
           
            # running eigen vector with largest eigen value estimate
            self.u = self.u.to(self.conv_layer.conv_layers[0].weight.device)
            u = self.u
            # power iterations
            for i in range(n_iter - 1):
                # normalization
                u = normalize(u)
                # W u
                u = self.conv_layer.forward(u)
                # D' W u
                if self.use_splines:
                    u = u * slope_max.view(1,-1,1,1)
                # WT D' W u
                u = self.conv_layer.transpose(u)
                # norm of u
                sigma_estimate = norm(u)

        # embdding the computation in the forward
        if differentiable:
            u = normalize(u)
            # W u
            u = self.conv_layer.forward(u)
            # D' W u
            if self.use_splines:
                slope_max = self.activation.slope_max
                u = u * slope_max.view(1,-1,1,1)
            
            # WT D' W u
            u = self.conv_layer.transpose(u)
            # norm of u
            sigma_estimate = norm(u)
            # update running estimate
            self.u = u
            return(sigma_estimate)
        else:
            # update running estimate
            self.u = u
            return(sigma_estimate)

    @property
    def device(self):
        return(self.conv_layer.conv_layers[0].weight.device)
    
    def prune(self, tol=1e-4, prune_filters=True, collapse_filters=False, change_splines_to_clip=False):
        """Prune the model by (only for testing):
            - removing filters with small weights/almost vanishing activations
            - collapsing the remaining filters into a single convolution
            - changing the splines to clipped linear functions
            
            These changes improve the computational efficiency of the model but might alter a bit the performances"""
        
        device = self.conv_layer.conv_layers[0].weight.device

        if collapse_filters:
            # 1. Convert multi-convolutions into single convolutions
            # 1.1 size of the single kernel
            new_padding = sum([conv.kernel_size[0]//2 for conv in self.conv_layer.conv_layers])
            new_kernel_size = 2*new_padding + 1

            # 1.2 Find new kernels <=> impulse responses
            impulse = torch.zeros((1, 1, new_kernel_size , new_kernel_size), device=device, requires_grad=False)
            impulse[0, 0, new_kernel_size//2, new_kernel_size//2] = 1

            new_kernel = self.conv_layer.convolution(impulse)

           
            # 2. Collapse convolutions
            new_conv_layer = MultiConv2d(channels=[1, new_kernel.shape[1]], kernel_size=self.channels[-1], padding=new_padding)

            new_conv_layer.conv_layers[0].parametrizations.weight.original.data = new_kernel.permute(1, 0, 2, 3)

            self.conv_layer = new_conv_layer
            self.channels = [1, new_kernel.shape[1]]
            self.padding = new_padding

        if prune_filters:
            # 1. Remove non significant filters
            # 1.1 size of the single kernel
            new_padding = sum([conv.kernel_size[0]//2 for conv in self.conv_layer.conv_layers])
            new_kernel_size = 2*new_padding + 1

            # 1.2 Find new kernels <=> impulse responses
            impulse = torch.zeros((1, 1, new_kernel_size , new_kernel_size), device=device, requires_grad=False)
            impulse[0, 0, new_kernel_size//2, new_kernel_size//2] = 1

            new_kernel = self.conv_layer.convolution(impulse)

            # 2. Determine the channels to prune, based on
            #     - impulse response magnitude
            kernel_norm = torch.sum(new_kernel**2, dim=(0, 2, 3))

            #     - TV2 of associated activation function
            coeff = self.activation.projected_coefficients
            slopes = (coeff[:,1:] - coeff[:,:-1])/self.activation.grid.item()
            tv2 = torch.sum(torch.abs(slopes[:,1:-1]), dim=1)
            
            # criterion to keep a (filter, activation) tuple
            weight = tv2 * kernel_norm

            l_keep = torch.where(weight > tol)[0]
            print("---------------------")
            print(f" PRUNNING \n Found {len(l_keep)} filters with non-vanishing potential functions")
            print("---------------------")


            # 3. Prune spline coefficients
            new_spline_coeff = torch.clone(self.activation.coefficients_vect.view(self.activation.num_activations, self.activation.size)[l_keep, :].contiguous().view(-1))
            self.activation.coefficients_vect.data = new_spline_coeff
            self.activation.num_activations = len(l_keep)

            self.activation.grid_tensor = torch.linspace(-self.activation.range_, self.activation.range_, self.activation.size).expand((self.activation.num_activations, self.activation.size))

            self.activation.init_zero_knot_indexes()

            # 4. Prune convolutions
            self.conv_layer.conv_layers[-1].parametrizations.weight.original.data = self.conv_layer.conv_layers[-1].parametrizations.weight.original.data[l_keep, :, :, :].permute(0, 1, 2, 3)

            self.channels[-1] = len(l_keep)

        if change_splines_to_clip:
            self.activation = self.activation.get_clip_equivalent()
        # 5. Update number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f" Number of parameters after prunning: {self.num_params}")
    

    def get_derivative(self, x):
        """
        It computes pointwisely the evaluation of the second derivative of the quadratic splie (i.e. the first derivative of the linear spline) accordind to the channel of 
        the input.
        """
        with torch.no_grad():
            y = self.conv_layer(x)
            grid = self.activation.grid.to(self.activation.coefficients_vect.device)
            zero_knot_indexes = self.activation.zero_knot_indexes.to(grid.device)
            coefficients_vect = self.activation.projected_coefficients_vect
            size = self.activation.size
            even = self.activation.even
            max_range = (grid.item() * (size // 2 - 1))
            if even:
                y = y - grid / 2
                max_range = (grid.item() * (size // 2 - 2))
            y_clamped = y.clamp(min=-(grid.item() * (size // 2)), max=max_range)

            floored_y = torch.floor(y_clamped / grid)  #left coefficient

            # This gives the indexes (in coefficients_vect) of the left
            # coefficients
            indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_y).long()

            # Only two B-spline basis functions are required to compute the output
            # (through linear interpolation) for each input in the B-spline range.
            activation_output = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / grid.item()
            # It avoid to store some unuseful variables on the restricted memory of the gpu
            del coefficients_vect
            del indexes
            del floored_y
            del y_clamped
            torch.cuda.empty_cache()
            return activation_output
    
    def set_W_Conv(self, n_heigth, n_width):
        """
        It sets the matrix W (when the image is vectorized, we can see the convolutional neural
        network as matrix-vector product) to the attribute of the model
        """
        dim = n_heigth*n_width
        device=self.conv_layer.conv_layers[0].weight.device
        output = torch.empty(dim, 1, n_heigth, n_width)
        for i in range(dim):
            a = torch.zeros(dim)
            a[i] = 1.
            output[i,:,:,:] = a.view(n_heigth, n_width)
        self.W_Conv = self.conv_layer(output.to(device))
        del output
        torch.cuda.empty_cache() 

    def Hessian(self, x):
        """
        It computes the Hessian of the model evaluated at x
        """
        n_heigth = x.shape[1]
        n_width = x.shape[2]
        derivative_x = self.get_derivative(x)
        if self.W_Conv is None:
            self.set_W_Conv(n_heigth=n_heigth, n_width = n_width)
        torch.cuda.empty_cache()
        return self.conv_layer.transpose(kronecker_prod(self.W_Conv, derivative_x))






        

def norm(u):
    return(torch.sqrt(torch.sum(u**2)))

def normalize(u):
    return(u/norm(u))

def kronecker_prod(W, A):
    dim = W.shape[0]
    output = torch.empty_like(W)
    for i in range(dim):
        output[i,:,:,:] = W[i,:,:,:] * A
    return output