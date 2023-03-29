import torch
import numpy as np
import matplotlib.pyplot as plt

class InversePhysicsInformedBarModel:
    """
    A class used for the definition of the data driven approach for Physics Informed Models for one dimensional bars. 
    EA is estimated.
    """

    def __init__(self, x, u, L, dist_load):
        """Construct a InversePhysicsInformedBarModel model"""

        '''
         Enter your code
         Task : initialize required variables for the class
        '''
        self.x = x
        self.u = u
        self.L = L
        self.dist_load = dist_load

        # defining the layers of the model

        layer1 = torch.nn.Linear(1, 20)
        activation1 = torch.nn.Tanh()
        layer2 = torch.nn.Linear(20, 1)

        # defining the model

        ea = torch.nn.Sequential(
            layer1,
            activation1,
            layer2,
        )

        self.ea = ea

    def predict(self, x, u):
        """Predict parameter EA of the differential equation."""

        '''
        Params: 
            x - input spatial value
            u - input displacement value at x
            ea - model predicted value
        '''

        '''
        Enter your code
        '''
        
        ea = self.ea(x)

        return ea

    def cost_function(self, x, u, EA_pred):
        """Compute the cost function."""

        '''
        Params:
            x - input spatial value
            u - displacement value at x
            EA_pred - model predicted EA value
            differential_equation_loss - calculated physics loss
        '''

        '''
        Enter your code
        '''

        EA_pred_x = torch.autograd.grad(EA_pred, x, grad_outputs=torch.ones_like(EA_pred), create_graph=True, retain_graph=True)[0]
    
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        differential_equation_loss = torch.mean((EA_pred*u_xx + EA_pred_x*u_x + self.dist_load(x))**2)

        return differential_equation_loss
    
    def train(self, epochs, optimizer, **kwargs):
        """Train the model."""

        '''
        This function is used for training the network. While updating the model params use "cost_function" 
        function for calculating loss
        Params:
            epochs - number of epochs
            optimizer - name of the optimizer
            **kwarhs - additional params

        This function doesn't have any return values. Just print the losses during training
        
        '''

        '''
            Enter your code        

        '''

        self.loss_history = []
        optimizer = torch.optim.LBFGS(self.ea.parameters(), lr=0.01)
        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                EA_pred = self.predict(self.x, self.u)
                loss = self.cost_function(self.x, self.u, EA_pred)
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)
            loss = closure()
            self.loss_history.append(loss.item())
            if epoch % 20 == 0:
                print(f'epoch {epoch}/{epochs}, Total loss {loss.item()}')