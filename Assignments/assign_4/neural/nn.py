# Here we will write the classes for the neural network

import numpy as np
import pandas as pd


class NeuralNetwork:

    """
    A class to represent a neural network.

    Attributes:
    - layers (numpy.ndarray):
        A 1D array of Layer objects representing the layers of the neural network.

    - loss:
        A function which takes two arguments: y_true and y_pred.

    - loss_prime:
        A function which takes two arguments: y_true and y_pred.

    Methods:
    --------

    - add(layer):
        Adds a layer to the neural network.

    - use_loss(loss):
        Sets the loss function and its derivative to be used by the neural network.

    - predict(X):
        Predicts the output of the neural network.

    - fit(X, y, epochs, learning_rate):
        Trains the neural network.
    """

    def __init__(self):
        self.layers = np.array([])
        self.loss = None
        self.loss_prime = None

    def get_dense_layers_w(self):
        l1_w = 0
        l2_w = 0

        # calculating the regularization term for weights of all the dense layers at once
        for layer in reversed(self.layers):
            if layer.__class__.__name__ == "DenseLayer":
                if layer.l1 != 0:
                    l1_w += np.sum(np.abs(layer.weights))
                if layer.l2 != 0:
                    l2_w += np.sum(np.power( layer.weights, 2))

        return l1_w, l2_w

    # add layer to network
    def add(self, layer):

        """
        Adds a layer to the neural network.

        Parameters:
        - layer (Layer): The Layer object to be added to the neural network.
        """

        self.layers = np.append(self.layers, layer)

    # set loss to use
    def use_loss(self, loss):

        """
        Sets the loss function and its derivative to be used by the neural network.

        Parameters:
        - loss (callable):
            A class which have two methods: loss and loss_prime.
                - that takes two arguments: y_true and y_pred.
                  shape = (n_classes, batch_size)

        - regularization (optional):
            "l1" or "l2" or None, default None

        """

        self.loss = loss.loss
        self.loss_prime = loss.loss_prime

        # adding the regularization for all the dense layers at once, it can be done for each layer separately as well.

        if loss.l1 != 0 or loss.l2 != 0:
            for layer in reversed(self.layers):
                if layer.__class__.__name__ == "DenseLayer":
                    # adding the regularization to the layer only if it is not already added while adding the layer
                    if layer.l1 != 0:
                        layer.l1 = loss.l1
                    if layer.l2 != 0:
                        layer.l2 = loss.l2

    # predict output for given input
    def predict(self, input_data):

        """
        Predicts the output for a given input.

        Parameters:

        - input_data (numpy.ndarray):
            A 2D array of shape (m_samples, n_features) representing m_samples, each with n_features.

        Returns:

        - result (numpy.ndarray):
            A list of 2D arrays of shape (m_samples, n_classes) representing the predicted output for each of the m_samples. n_classes is the number of output nodes in the neural network.
        """

        # run network over all samples
        output = input_data.T
        for layer in self.layers:
            # if layer is "DropoutLayer": then pass
            if layer.__class__.__name__ == "DropoutLayer":
                continue
            output = layer.forward_propagation(output)

        return output.T
    
    def summary(self):
        total_params = 0
        print("Summary of the Neural Network")
        print("_"*115)
        print("Layer (type)".ljust(20) + "Neurons #".ljust(15) + "Input Shape".ljust(15) +"Output Shape".ljust(15)+ "Weights shape".ljust(15) + "Bias shape".ljust(15) + "Param #".rjust(10))
        print("="*115)
        # print for input
        print("Input".ljust(20) + f"{self.layers[0].weights.shape[1]}".ljust(15)+ f"-".ljust(15) + f"{self.layers[0].weights.shape[1], None}".ljust(15) + "-".ljust(15)+ "-".ljust(15)+ "0".rjust(10))
        print()
        for layer in self.layers:
            layer_name = layer.__class__.__name__
            
            if layer_name == "ActivationLayer":
                print(f"{layer_name}".ljust(20) +
                    f"-".ljust(15) +
                    f"activavtion".ljust(15) +
                    f"{layer.activation_name}".ljust(15) +
                    f"-".ljust(15) +
                    f"-".ljust(15) +
                    f"0".rjust(10))
                print()
                continue
            if layer_name == "DropoutLayer":
                print(f"{layer_name}".ljust(20) +
                    f"-".ljust(15) +
                    f"droupout rate: ".ljust(15) +
                    f"{layer.dropout_rate}".ljust(15) +
                    f"-".ljust(15) +
                    f"-".ljust(15) +
                    f"{layer.trainable_params}".rjust(10))
                print()
                continue
            if layer_name == "BatchNormalization":
                print(f"{layer_name}".ljust(20) +
                    f"-".ljust(15) +
                    f"epsilon: ".ljust(15) +
                    f"{layer.epsilon}".ljust(15) +
                    f"-".ljust(15) +
                    f"-".ljust(15) +
                    f"-".rjust(10))
                print()
                continue
            
            print(f"{layer_name}".ljust(20) + 
                f"{layer.nuerons}".ljust(15) + 
                f"{layer.weights.shape[1], None}".ljust(15) +
                f"{layer.nuerons,None}".ljust(15) +
                f"{layer.weights.shape}".ljust(15) +
                f"{0 if isinstance(layer.biases, int) else layer.biases.shape}".ljust(15) +
                f"{layer.trainable_params}".rjust(10))
            print()
            total_params += layer.trainable_params

            

        print("="*115)
        print("Total params".ljust(20) + f"{total_params}".rjust(85))


        # print("Trainable params: {}".format(self.get_trainable_params_count()))


    # for verbose
    def __verbose(self,  y_train, x_train, iter, batch_iter, verbo):

        if verbo !=0:
            ws = self.get_dense_layers_w()

            if verbo ==3:
                if (batch_iter) % 50 == 0:
                    print(f"Epoch {iter+1}- Batch {batch_iter+1}-{self.batches}", end=" ======================> ")

                    print(f"cost: {self.loss(y_train.T, self.predict(x_train).T, ws):.4f}")

            elif verbo == 2 and batch_iter ==-1:
                    print(f"Epoch {iter+1}-{self.epochs}", end=" ======================> ")
                    print(f"cost: {self.loss(y_train.T, self.predict(x_train).T, ws):.4f}")

            elif batch_iter ==-1:
                if iter % 10 == 0:
                    print(f"Epoch {iter+1}-{self.epochs}", end=" ======================> ")
                    print(f"cost: {self.loss(y_train.T, self.predict(x_train).T, ws):.4f}")


    # train the network
    def fit(
        self, x_train:np.ndarray, y_train: np.ndarray, epochs:int, learning_rate=0.01, batch_size=64, verbose=1, callback=None
    ):

        """
        Trains the neural network on the training data.

        Parameters:
        - x_train (numpy.ndarray):
            A 2D array of shape (m_samples, n_features) representing m_samples, each with n_features.

        - y_train (numpy.ndarray):
            A 2D array of shape (m_samples, 1) representing m_samples each with n_classes outputs.
            one-hot encoded for classification

        - epochs (int):
            The number of training iterations to perform.

        - learning_rate (float):
            The learning rate to use for gradient descent.

        - batch_size (int): default 64
            The number of samples to use in each batch. If None, the entire training set is used in each batch.

        - verbose (int): default 1
            0: no output
            1: print loss after 10th epoch
            2: print loss after each epoch
            3: print loss after each epoch for every 50th batch

        - callback (function): default None
            A function to call after each epoch. The function should take the current epoch as an argument.

        """

        # if n_classes > 1:
        #     # one-hot encode the labels
        #     y_train = pd.get_dummies(y_train).values

        # if no batch size is provided, use the entire training set in each batch
        if (batch_size is None) or (batch_size > x_train.shape[0]):
            batch_size = x_train.shape[0]

        # number of batches
        batches = x_train.shape[0] // batch_size

        if batches == 0:
            batches = 1

        # train network for given number of epochs
        self.epochs = epochs
        self.batches = batches
        for i in range(self.epochs):

            # shuffle training data
            idx = np.random.permutation(x_train.shape[0])
            x_train = x_train[idx]
            y_train = y_train[idx]
            # train network on all batches
            for j in range(self.batches):
                batch_start, batch_end = j * batch_size, (j + 1) * batch_size

                # forward propagation
                output = x_train[batch_start:batch_end].T  # shape (nx , batch_size)
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # backward propagation
                error = self.loss_prime(
                    y_train[batch_start:batch_end].T, output
                )  # shape (ny , batch_size)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

                 # Print verbose messages 
                self.__verbose(y_train, x_train, iter=i, batch_iter=j, verbo=verbose)

            if callback is not None:
                pred = self.predict(x_train)
                # send the normal y_train and not the one-hot encoded
                true = np.argmax(y_train, axis=1)
                loss_ = self.loss(y_train.T, pred.T, self.get_dense_layers_w())
                callback(true, pred, loss_, i)


            # Print verbose messages
            self.__verbose(y_train, x_train, iter=i, batch_iter=-1, verbo=verbose)
        
        # for last epoch
        self.__verbose(y_train, x_train, iter=self.epochs-1, batch_iter=-1, verbo=1)

