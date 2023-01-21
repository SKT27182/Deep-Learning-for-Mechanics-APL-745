# File to calculate Gradiant Descent

import numpy as np

class BaseGD:
    def __init__(self, alpha, max_iter,tol=None, bias=False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = None
        self.weights = None
        self.bias = bias
        self.loss_history = []


    def _preprocess_input_data(self,X):
        
        # if input data is not numpy.ndarray then  convert it
        if isinstance(X,np.ndarray):
            pass
        else:
            X = np.array(X)

        # if only one sample is given then reshape it
        if X.ndim  == 1:
            X = X.reshape(1, -1)
        
        return X.astype(np.float64)



    def _bias(self,X):
        if self.bias:
            if len(X.shape) == 1:
                X = X.reshape(1,-1)
            # add bias term to X (w0*x0)
            return np.insert(X, 0, 1., axis=1)
        else:
            return X


    def _weights(self, n):
        if self.bias:
            return np.random.random(n + 1)
        else:
            return np.random.random(n)

    def _y_hat(self, X, w):
        return np.dot(X, w.T)
        

    def _cal_loss(self, y_hat, y_true):
        """

        Calculate the cost function to check convergence

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        y : numpy.ndarray , shape (m_samples, )
            Target values

        w : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        Returns
        -------
        cost : float
            Cost

        """


        # no. of training examples
        m = len(y_true)

        # initialize loss
        total_loss = 0

        # calculate cost
        # y_hat = self._y_hat(X, w)
        total_cost = np.sum(np.square(y_hat - y_true))
        # return cost
        return total_cost / (2 * m)


    def _cal_gradient(self, y_hat ,y_true , X):
        """

        Calculate the gradient of the cost function to update the weights

        Parameters
        ----------
        y_hat : numpy.ndarray , shape (m_samples, )
            Calculated value of y

        y_true : numpy.ndarray , shape (m_samples, )
            True value of y

        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        Returns
        -------
        gradient : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Gradient

        """

        # no. of training examples
        m = len(y_true)

        gradient = np.matmul((y_hat- y_true), X) / m

        # return gradient
        return gradient 

    def predict(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        Returns
        -------
        y_hat : numpy.ndarray , shape (m_samples, )
            Predicted values

        """
        if self.weights is None:
            raise AttributeError("You have to fit the model first")
        
        X = self._preprocess_input_data(X)
        X = self._bias(X)
        return self._y_hat(X,self.weights) 



    

class BatchGD(BaseGD):

    def __init__(self, alpha, max_iter, bias=False, tol=None):
        super().__init__(alpha=alpha, max_iter=max_iter, bias=bias, tol=tol)


    def fit(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values
        alpha : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        tol : float, default None
            Tolerance for stopping criteria

        Returns
        -------
        w : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        """

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # to include the bias term or not and initialize weights
        X = self._bias(X)
        self.weights = self._weights(n)


        # iterate until max_iter

        for i in range(self.max_iter):

            #calculate the gradient of Loss/Cost Function

            y_hat = self._y_hat(X, self.weights)
            gradient = self._cal_gradient(y_hat ,y , X)

            # update the weights
            self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            self.loss_history.append(self._cal_loss(y_hat, y))

            # Break the loop if loss is not changing much
            if len(self.loss_history) > 1 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break




class MiniBatchGD(BaseGD):
    def __init__(self, alpha, max_iter, batch_size, bias=False, tol=None):
        super().__init__(alpha=alpha, max_iter=max_iter, bias=bias, tol=tol)
        self.batch_size = batch_size

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values
        alpha : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        tol : float, default None
            Tolerance for stopping criteria

        Returns
        -------
        w : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        """

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # to include the bias term or not and initialize weights
        X = self._bias(X)
        self.weights = self._weights(n)



        for i in range(self.max_iter):

            # shuffle the data
            shuffle_indices = np.random.permutation(np.arange(X.shape[0]))
            X_shuffle = X[shuffle_indices]
            y_shuffle = y[shuffle_indices]

            # split the data into batches
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffle[i:i + self.batch_size]
                y_batch = y_shuffle[i:i + self.batch_size]

                #calculate the gradient of Loss/Cost Function
                y_hat = self._y_hat(X_batch, self.weights)
                gradient = self._cal_gradient(y_hat ,y_batch , X_batch)

                # update the weights
                self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            self.loss_history.append(self._cal_loss(y_hat, y_batch))

            # Break the loop if loss is not changing much
            if len(self.loss_history) > 1 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break



class StochasticGD(MiniBatchGD):
    def __init__(self, alpha, max_iter, bias=False, tol=None):
        super().__init__(alpha=alpha, max_iter=max_iter,batch_size=1, bias=bias, tol=tol)
        
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values
        alpha : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        tol : float, default None
            Tolerance for stopping criteria

        Returns
        -------
        w : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        """

        # Stochastic Gradient Descent is 
        # same as MiniBatch Gradient Descent only different is that 
        # in Stochastich Gradient descent we take batch_size=1

        super().fit(X,y)
    


class LogisticRegression(StochasticGD):
    def __init__(self, alpha, max_iter, bias=False, tol=None):
        super().__init__(alpha=alpha, max_iter=max_iter, bias=bias, tol=tol)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cal_loss(self, X, y, w):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values
        w : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        Returns
        -------
        cost : float
            Cost

        """
        # no. of samples
        m = X.shape[0]

        # calculate cost
        cost = -1 / m * (np.dot(y.T, np.log(self.sigmoid(np.dot(X, w)))) + np.dot((1 - y).T, np.log(1 - self.sigmoid(np.dot(X, w)))))

        return cost

    def _cal_gradient(self, X, y, w):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values
        w : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        Returns
        -------
        gradient : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Gradient

        """
        # no. of samples
        m = X.shape[0]

        # calculate gradient
        gradient = 1 / m * np.dot(X.T, self.sigmoid(np.dot(X, w)) - y)

        return gradient

    def predict(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        """
        # no. of samples
        m = X.shape[0]

        # initialize y_pred
        y_pred = np.zeros(m)

        # calculate y_pred
        y_pred = self.sigmoid(np.dot(X, self.weights))

        # convert to 0 or 1
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0