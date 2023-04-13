"""
Solution of 2D Forward Problem of Linear Elasticity
   for Plane Stress Boundary Value Problem using
      Physics-Informed Neural Networks (PINN)
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

torch.manual_seed(123456)
np.random.seed(123456)

E = 1                                       # Young's Modulus
nu = 0.3                                    # Poisson Ratio
G = E/(2*(1+nu))                            # Shear Modulus

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define your model here (refer: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
        self.Nnet = nn.Sequential()
        self.Nnet.add_module('Linear_layer_1', nn.Linear(in_features=2, out_features=30))    # First linear layer
        self.Nnet.add_module('Tanh_layer_1', nn.Tanh())         # Add activation
        self.Nnet.add_module('Linear_layer_2', nn.Linear(in_features=30,out_features=30))    # Second linear layer
        self.Nnet.add_module('Tanh_layer_2', nn.Tanh())         # Add activation
        self.Nnet.add_module('Linear_layer_3', nn.Linear(in_features=30,out_features=30))    # Third linear layer
        self.Nnet.add_module('Tanh_layer_3', nn.Tanh())         # Add activation
        self.Nnet.add_module('Linear_layer_4', nn.Linear(in_features=30,out_features=2))     # Fourth linear layer
        self.Nnet.add_module('Tanh_layer_4', nn.Tanh())         # Add activation
        self.Nnet.add_module('Linear_layer_5', nn.Linear(in_features=2,out_features=2))      # Fifth linear layer
        self.Nnet.add_module('Tanh_layer_5', nn.Tanh())         # Add activation

        
        print(self.Nnet)                                        # Print model summary

    # Forward Feed
    def forward(self, x):
        y = self.Nnet(x)
        return y

    # PDE and BCs loss
    def loss(self, xy_f_train, xy_b_train, u_b_train, v_b_train):
        y = self.Nnet(xy_f_train)           # Interior Solution (output from from defined NN model)
        y_b = self.Nnet(xy_b_train)         # Boundary Solution (output from from defined NN model)
        u_b, v_b = y_b[:,0], y_b[:,1]       # Extract u and v from boundary solution
        u,v =  y[:,0], y[:,1]               # Extract u and v from interior solution


        # Calculate Gradients
        # Gradients of deformation in x-direction (first and second derivatives)
        u_g = torch.autograd.grad(u, xy_f_train, grad_outputs=torch.ones_like(u),allow_unused=True, create_graph=True)[0] # Gradient of u, Du = [u_x, u_y]
        u_x, u_y =  u_g[:,0], u_g[:,1]      # [u_x, u_y]
        u_xx =  torch.autograd.grad(u_x, xy_f_train, grad_outputs=torch.ones_like(u_x),allow_unused=True, create_graph=True)[0][:,0] # Second derivative, u_xx
        u_xy =  torch.autograd.grad(u_x, xy_f_train, grad_outputs=torch.ones_like(u_x),allow_unused=True, create_graph=True)[0][:,1] # Mixed partial derivative, u_xy
        u_yy =  torch.autograd.grad(u_y, xy_f_train, grad_outputs=torch.ones_like(u_y),allow_unused=True, create_graph=True)[0][:,1] # Second derivative, u_yy

        # Gradients of deformation in y-direction (first and second derivatives)
        v_g =   torch.autograd.grad(v, xy_f_train, grad_outputs=torch.ones_like(v),allow_unused=True, create_graph=True)[0] # Gradient of v, Dv = [v_x, v_y]
        v_x, v_y =  v_g[:,0], v_g[:,1]      # [v_x, v_y]
        v_xx =   torch.autograd.grad(v_x, xy_f_train, grad_outputs=torch.ones_like(v_x),allow_unused=True, create_graph=True)[0][:,0] # Second derivative, v_xx
        v_xy =  torch.autograd.grad(v_x, xy_f_train, grad_outputs=torch.ones_like(v_x),allow_unused=True, create_graph=True)[0][:,1] # Mixed partial derivative, v_xy
        v_yy =  torch.autograd.grad(v_y, xy_f_train, grad_outputs=torch.ones_like(v_y),allow_unused=True, create_graph=True)[0][:,1] # Second derivative, v_yy

        f_1 = G*[u_xx + u_yy] + G((1+nu)/(1-nu))*[u_xx + v_xy] + torch.sin(2*torch.pi*xy_f_train[:,0])*torch.sin(2*torch.pi*xy_f_train[:,1])                              # Define body force for PDE-1
        f_2 = G*[v_xx + v_yy] + G((1+nu)/(1-nu))*[v_xx + u_xy] + torch.sin(torch.pi*xy_f_train[:,0])*torch.sin(2*torch.pi*xy_f_train[:,1])                              # Define body force for PDE-2
        
        loss_1 = torch.mean((f_1)**2)       # Define loss for PDE-1
        loss_2 = torch.mean((f_2)**2)       # Define loss for PDE-2

        loss_PDE =  loss_1 + loss_2          # Total PDE loss
        loss_bc =    torch.mean((u_b - u_b_train)**2) + torch.mean((v_b - v_b_train)**2) # Total BC loss

        TotalLoss = loss_PDE + loss_bc       # Total loss
        print(f'epoch {epoch}: loss_pde {loss_PDE:.8f}, loss_bc {loss_bc:.8f}')
        return TotalLoss

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),allow_unused=True, create_graph=True)

# Define model parameters here
device = ("cuda" if torch.cuda.is_available() else "cpu")



# Load the collocation point data
xy_f_train = np.load('Lab Notes/Lab 11/interior_points.mat')
xy_b_train = np.load('Lab Notes/Lab 11/boundary_points.mat')


## Define data as PyTorch Tensor and send to device
xy_f_train = torch.tensor( xy_f_train, requires_grad=True, dtype=torch.float32).to(device)
xy_b_train = torch.tensor( xy_b_train, requires_grad=True, dtype=torch.float32).to(device)

# Define the boundary condition values
u_b_train = 
v_b_train = 

# Initialize model
model = Model().to(device)

# Loss and Optimizer
optimizer = 

# Training
def train(epoch):
    model.train()

    def closure():
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    print(f'epoch {epoch}: loss {loss:.8f} ')
    return loss

for epoch in range():
    train(epoch)

# %%
''' Testing PINN '''
# Generate collocation points and use trained model for testing

    
# %%
''' Plotting '''
