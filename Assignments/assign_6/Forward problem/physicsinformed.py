import torch
import numpy as np

class PhysicsInformedBarModel:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, x, E, A, L, u0, dist_load):
        """Construct a PhysicsInformedBar model"""

        '''
         Enter your code
         Task : initialize required variables for the class
        '''
        self.x = x
        self.E = E
        self.A = A
        self.L = L
        self.u0 = u0
        self.dist_load = dist_load


        # define the model

        layer1 = torch.nn.Linear(1, 40)
        activation1 = torch.nn.Tanh()
        layer2 = torch.nn.Linear(40, 40)
        activation2 = torch.nn.Tanh()
        layer3 = torch.nn.Linear(40, 1)    # output layer

        # defining the model

        u = torch.nn.Sequential(
            layer1,
            activation1,
            layer2,
            activation2,
            layer3,
        )

        self.u = u


    def costFunction(self, x, u_pred): 
        """Compute the cost function."""

        '''
        This function takes input x and model predicted output u_pred to compute loss
        Params:
            x - spatial value
            u_pred - NN predicted displacement value
            differential_equation_loss - calculated PDE residual loss
            boundary_condition_loss - calculated boundary loss
        '''
        
        '''
            Enter your code
        '''

        # boundary conditions
        boundary_u = torch.tensor([self.u0, 0.0])
        boundary_x = torch.tensor([0.0, self.L])

        # calculate boundary loss
        boundary_condition_loss = ((u_pred[0] - boundary_u[0])**2 + ((u_pred[-1] - boundary_u[1])**2))/2

        # calculate PDE residual loss
        ux = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
        uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]

        uxx += 4*torch.pi**2*torch.sin(2*torch.pi*x)

        differential_equation_loss = torch.mean((self.E * self.A * uxx )**2)

        return differential_equation_loss, boundary_condition_loss
    
    def get_displacements(self, x):
        """Get displacements."""

        '''
        This function is used while inference (you can even use in your training phase if needed.
        It takes x as input and returns model predicted displacement)
        Params:
            x - input spatial value
            u - model predicted displacement
        '''

        "Enter your code"
        
        return self.u(x)

    def train(self, epochs, optimizer, **kwargs):
        """Train the model."""

        '''
        This function is used for training the network. While updating the model params use "costFunction" 
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

        optimizer = torch.optim.Adam(self.u.parameters(), lr=0.001)
        self.loss_history = []


        for epoch in range(epochs):

            optimizer.zero_grad()
            u_pred = self.get_displacements(self.x)
            differential_equation_loss, boundary_condition_loss = self.costFunction(self.x, u_pred)
            loss = differential_equation_loss + boundary_condition_loss
            loss.backward()
            optimizer.step()

            self.loss_history.append([differential_equation_loss.item(), boundary_condition_loss.item(), loss.item()])

            if epoch % 100 == 0:
                print('epoch {}, loss {}'.format(epoch, loss.item()))

        


                