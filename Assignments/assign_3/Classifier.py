# File to calculate Gradiant Descent

import numpy as np

from scipy.optimize import minimize


class BaseGD:
    def __init__(self, alpha=0.01, max_iter=1000, bias=False, tol=None, penalty=None, lambda_=0.6 ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.bias = bias
        self.tol = tol
        self.penalty = penalty
        self.lambda_ = lambda_

        self.weights = None    
        self.loss_history = []

    """
    Abstract class for Gradient Descent

    Attributes
    ----------
    alpha : float
        Learning rate, default is 0.01

    max_iter : int
        Maximum number of iterations, default is 1000

    bias : bool
        If True then add bias term to X, default is False

    tol : float
        Tolerance for the cost function, default is None   
        If None then it will run for max_iter

    penalty : str
        Type of regularization, default is None

    lambda_ : float
        Regularization parameter, default is 0.6

    weights : numpy.ndarray
        Weights of the model

    loss_history : list
        List of cost function values for each iteration

    Methods
    -------

    fit(X, y)
        Fit the model

    predict(X)
        Predict the values

    r2_score(X, y)
        Calculate the r2 score

    """

    def _preprocess_input_data(self, X):
        """
        Convert input data to numpy.ndarray and reshape it if needed

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        Returns
        -------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data after preprocessing like converting to numpy.ndarray and reshaping
        """
        # if input data is not numpy.ndarray then  convert it
        if isinstance(X, np.ndarray):
            pass
        else:
            X = np.array(X)

        # if only one sample is given then reshape it
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X.astype(np.float64)

    def _bias(self, X):
        """
        Add bias term to X

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        Returns
        -------
        X : numpy.ndarray , shape (m_samples, n_features + 1 )
            Testing data with bias term
        """
        if self.bias:
            # add bias term to X (w0*x0)
            return np.insert(X, 0, 1., axis=1)
        else:
            return X

    def _weights(self, n):

        """
        Initialize weights

        Parameters
        ----------
        n : int
            Number of features

        Returns
        -------
        weights : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        """
        if self.bias:
            return np.random.uniform(low=-1, high=1, size=n +1)
            # return np.random.random(n+1)

        else:
            return np.random.uniform(low=-1, high=1, size=n)
            # return np.random.random(n)

    def _y_hat(self, X, w):

        """
        Calculate the predicted value of y

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        w : numpy.ndarray , shape (n_features + 1 ) +1 for bias
            Weights

        Returns
        -------
        y_hat : numpy.ndarray , shape (m_samples, )
            Calculated value of y
        """
        return np.dot(X, w.T)

    def _cal_loss(self, y_hat, y_true):
        """
        
        Calculate the cost function
        
        Parameters
        ----------
        y_hat : numpy.ndarray , shape (m_samples, )
            Calculated value of y
            
        y_true : numpy.ndarray , shape (m_samples, )
            True value of y
            
        Returns
        -------
        loss : float
            Loss
            
        """

        # no. of training examples
        m = len(y_true)

        # initialize loss
        total_cost = 0

        # calculate cost
        # y_hat = self._y_hat(X, w)
        total_cost = np.sum(np.square(y_hat - y_true))/(2*m)

        if self.penalty is None:
            return total_cost
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sum(np.abs(self.weights))
            return total_cost + regularization
        elif self.penalty == "l2":
            regularization = (self.lambda_ / (2*m)) * np.sum(self.weights**2)
            return total_cost + regularization
        else:
            raise ValueError("Invalid penalty type")

    def _cal_gradient(self, y_hat, y_true, X):
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

        gradient = np.matmul((y_hat - y_true), X) / m

        if self.penalty is None:
            return gradient
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * \
                np.sign(self.weights)
            return gradient + regularization
        elif self.penalty == "l2":
            regularization = (self.lambda_ / m) * self.weights
            return gradient + regularization
        else:
            raise ValueError("Invalid penalty type")


    def predict(self, X):
        """

        Predict the values of y for the given X

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        """
        if self.weights is None:
            raise AttributeError("You have to fit the model first")

        X = self._preprocess_input_data(X)
        X = self._bias(X)
        return self._y_hat(X, self.weights)

    def r2_score(self, X, y):
        """

        Calculate the R2 score

        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Testing data

        y : numpy.ndarray , shape (m_samples, )
            True values

        Returns
        -------
        score : float
            R2 score

        """
        y_hat = self.predict(X)
        SSres = np.sum((y - y_hat)**2)
        SStot = np.sum((y - y.mean())**2)

        return 1 - (SSres / SStot)


class BatchGD(BaseGD):

    def __init__(self, alpha, max_iter, bias=False, tol=None, penalty=None, lambda_=0.6 ):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)




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
        None, just updates the weights and loss_history attributes of the class

        """

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # to include the bias term or not and initialize weights
        X = self._bias(X)
        self.weights = self._weights(n)

        # iterate until max_iter

        for i in range(self.max_iter):

            # calculate the gradient of Loss/Cost Function

            y_hat = self._y_hat(X, self.weights)
            gradient = self._cal_gradient(y_hat, y, X)

            # update the weights
            self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            y_hat = self._y_hat(X, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y))

            # Break the loop if loss is not changing much
            if i > 0 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(
                    f"Loss is nan at iteration {i}. Hence, stopping the training")
                break


class MiniBatchGD(BaseGD):
    def __init__(self, alpha, max_iter, bias=False, tol=None, penalty=None, lambda_=0.6 ):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)


    def fit(self, X, y, batch_size=32):

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
        None, just updates the weights and loss_history attributes of the class

        """

        self.batch_size = batch_size

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # no. of samples
        m = X.shape[0]

        # to include the bias term or not and initialize weights
        X = self._bias(X)
        self.weights = self._weights(n)

        # converting y to numpy array b/c if it is dataframe then it will give error while creating batches
        y = np.array(y)

        for i in range(self.max_iter):

            # shuffle the data
            shuffle_indices = np.random.permutation(np.arange(m))
            X_shuffle = X[shuffle_indices]
            y_shuffle = y[shuffle_indices]

            # split the data into batches
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffle[i:i + self.batch_size]
                y_batch = y_shuffle[i:i + self.batch_size]

                # calculate the gradient of Loss/Cost Function
                y_hat = self._y_hat(X_batch, self.weights)
                gradient = self._cal_gradient(y_hat, y_batch, X_batch)

                # update the weights
                self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            y_hat = self._y_hat(X, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y_batch))

            # Break the loop if loss is not changing much
            if i > 1 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(
                    f"Loss is nan at iteration {i}. Hence, stopping the training")
                break


class StochasticGD(BaseGD):
    def __init__(self, alpha, max_iter, bias=False, tol=None, penalty=None, lambda_=0.6 ):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)


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
        None, just updates the weights and loss_history attributes of the class

        """

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # no. of samples
        m = X.shape[0]

        # to include the bias term or not and initialize weights
        X = self._bias(X)
        self.weights = self._weights(n)

        # converting y to numpy array b/c if it is dataframe then it will give error while creating batches
        y = np.array(y)

        for i in range(self.max_iter):

            # Update the weights one by one for each data point
            for i in range(0, m):

                # take a data point randomly at a time and update the weights
                index = np.random.randint(m)

                X_ = X[index:index + 1]
                y_ = y[index:index + 1]

                # calculate the gradient of Loss/Cost Function
                y_hat = self._y_hat(X_, self.weights)
                gradient = self._cal_gradient(y_hat, y_, X_)

                # update the weights
                self.weights = self.weights - (self.alpha * gradient)

            # keeping track of cost/loss
            y_hat = self._y_hat(X_, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y_))

            # Break the loop if loss is not changing much
            if i > 1 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(f"Loss is nan at iteration {i}. Hence, stopping the training")
                break


class LogisticRegression(BatchGD):
    def __init__(self, alpha, max_iter, bias=False, tol=None,  penalty=None, lambda_=0.6 ):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)

    def _sigmoid(self, z):
        """
        Parameters
        ----------
        z : numpy.ndarray , shape (m_samples, )
            Input values
            
        Returns
        -------
        y_hat : numpy.ndarray , shape (m_samples, )
            Predicted values
        
        """

        return 1 / (1 + np.exp(-z))

    def _y_hat(self, X, w):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        w : numpy.ndarray , shape (n_features, )
            Weights

        Returns
        -------
        z : numpy.ndarray , shape (m_samples, )
            Linear fited values before applying sigmoid function 

        """
        return self._sigmoid(np.dot(X, w))

    def _cal_loss(self, y_hat, y_true):
        """
        Calculate the Binary Cross Entropy Loss

        Parameters
        ----------
        y_hat : numpy.ndarray , shape (m_samples, )
            Predicted values

        y_true : numpy.ndarray , shape (m_samples, )
            Target values

        Returns
        -------
        cost : float
            Cost

        """
        # no. of samples
        m = len(y_true)

        # calculate cost in term of y_true and y_hat
        total_cost = np.sum(-y_true * np.log(y_hat) - (1 - y_true) * (np.log(1 - y_hat))) / m

        if self.penalty is None:
            return total_cost
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sum(np.abs(self.weights))
            return total_cost + regularization
        elif self.penalty == "l2":
            regularization = (self.lambda_ / (2*m)) * np.sum(self.weights**2)
            return total_cost + regularization
        else:
            raise ValueError("Invalid penalty type")


    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted Probabilities

        """

        # preprocess the input data
        X = self._preprocess_input_data(X)

        # include the bias term or not
        X = self._bias(X)

        # calculate y_pred
        return self._y_hat(X, self.weights)

    def predict(self, X, threshold=0.5):

        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        threshold : float
            Threshold value

        Returns
        -------
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        """

        self.threshold = threshold

        y_pred = self.predict_proba(X)

        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0

        return y_pred

    def accuracy(self, y_true, y_pred):
        """
        Parameters
        ----------
        y_true : numpy.ndarray , shape (m_samples, )
            Target values
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        Returns
        -------
        accuracy : float
            Accuracy

        """
        return np.sum(y_true == y_pred)/len(y_true)


class OneVsAll(LogisticRegression):
    def __init__(self, alpha, max_iter, bias=False, tol=None,  penalty=None, lambda_=0.1):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)
        self.penalty = penalty
        self.lambda_ = lambda_


    def fit(self, X, y):

        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
            
        y : numpy.ndarray , shape (m_samples, )
        Target values
        
        Returns
        -------
        None, but updates the weights and loss history attributes
        """

        self.classes = np.unique(y)

        if len(self.classes) < 3:
            super().fit(X, y)
        else:
            if self.bias:
                self.all_weights = np.zeros((len(self.classes), X.shape[1]+1))
            else:
                self.all_weights = np.zeros((len(self.classes), X.shape[1]))

            for i in range(len(self.classes)):
                y_new = np.where(y == self.classes[i], 1, 0)
                super().fit(X, y_new)
                self.all_weights[i] = self.weights

    def predict_proba(self, X):

        if self.weights is None:
            raise AttributeError("You have to fit the model first")


        X = self._preprocess_input_data(X)
        X = self._bias(X)

        if hasattr(self, "all_weights"):
            # taken a transpose b/c earlier we were saving our weights in 1-d array
            # but now we are saving the weights in classes * features (classes,features), earlier it was like
            # feature * class (n_features,)
            self.weights = self.all_weights.T
        return super().predict_proba(X)

    def predict(self, X, threshold=0.5):

        if hasattr(self, "all_weights"):
            para_pred = self.predict_proba(X)
            return self.classes[np.argmax(para_pred, axis=1)]
        return super().predict(X, threshold=threshold)


class SoftmaxLogisticReg(BatchGD):
    def __init__(self, alpha, max_iter, bias=False, tol=None, penalty=None, lambda_=0.6, n_classes=2):
        super().__init__(alpha, max_iter, bias, tol, penalty, lambda_)
        self.n_classes = n_classes

    def _softmax(self, z):
        # stable softmax, to avoid overflow and underflow errors 
        # while calculating softmax for large values
        z -= np.max(z, axis=1, keepdims=True)
        exps = np.exp(z)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _y_hat(self, X, w):
        z = np.dot(X, w)
        return self._softmax(z)

    def _cal_loss(self, y_hat, y_true):
        m = y_true.shape[0]
        # calculate the loss function
        total_cost = -np.sum(y_true * np.log(y_hat+1e-10))/(2*m)
        # add the regularization term
        if self.penalty is None:
            return total_cost
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sum(np.abs(self.weights))
            return total_cost + regularization
        elif self.penalty == "l2":
            regularization = (self.lambda_ / (2*m)) * np.sum(self.weights**2)
            return total_cost + regularization
        else:
            raise ValueError("Invalid penalty type")

    # calculate gradient of the cross entropy loss function
    def _cal_gradient(self, y_hat, y_true, X):
        m = y_true.shape[0]
        gradient = np.dot(X.T, y_hat - y_true)
        # add the regularization term
        if self.penalty is None:
            return gradient
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sign(self.weights)
            return gradient + regularization
        elif self.penalty == "l2":
            regularization = (self.lambda_ / m) * self.weights
            return gradient + regularization
        else:
            raise ValueError("Invalid penalty type")

    def fit(self, X, y):

        X = self._preprocess_input_data(X)

        # no. of features
        n = X.shape[1]

        # to include the bias term or not and initialize weights
        X = self._bias(X)


        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(low=-1, high=1, size=(n_features, self.n_classes))
        
        y_encoded = np.zeros((n_samples, self.n_classes))
        for i in range(self.n_classes):
            y_encoded[:, i] = (y == self.classes[i]).astype(int)

        for i in range(self.max_iter):
            y_hat = self._y_hat(X, self.weights)
            # gradient = np.dot(X.T, y_encoded - y_hat)
            gradient = self._cal_gradient(y_hat, y_encoded, X)
            self.weights = self.weights - (self.alpha * gradient)
            

            y_hat = self._y_hat(X, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y_encoded))


            # Break the loop if loss is not changing much
            if i > 1 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(f"Loss is nan at iteration {i}. Hence, stopping the training")
                break

    def predict_proba(self, X):
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

        if self.weights is None:
            raise AttributeError("You have to fit the model first")

        # preprocess the input data
        X = self._preprocess_input_data(X)

        # include the bias term or not
        X = self._bias(X)
        
        y_hat = self._y_hat(X, self.weights)
        return y_hat

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

        y_hat = self.predict_proba(X)
        return np.argmax(y_hat, axis=1)

    

class LinearSearchGD(BaseGD):
    def __init__(self, max_iter, bias=False, tol=None,  penalty=None, lambda_=0.1):
        super().__init__(alpha=None, max_iter=max_iter, bias=bias, tol=tol, penalty=penalty, lambda_=lambda_)
        self.alpha_history = []

    """
    Linear Search Gradient Descent
    

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations
    bias : bool
        To include the bias term or not
    tol : float
        Tolerance for the stopping criterion
   
    Attributes
    ----------
    weights : numpy.ndarray , shape (n_features, )
        Weights after fitting the model
    loss_history : list
        Loss history after each iteration



    Methods
    -------
    fit(X, y)
        Fit the model according to the given training data.

    predict(X)
        Predict the y values for the given X.

    r2_score(y_true, y_pred)
        Calculate the R2 score.

    """

    def phi(self, X, y, weights, alpha):
        # calculate the loss function w.r.t alpha

        y_hat = self._y_hat(X, weights)
        grad = self._cal_gradient(y_hat, y, X)

        weights_new = weights - alpha * grad
        y_hat_new = self._y_hat(X, weights_new)

        return self._cal_loss(y_hat_new, y)

    def phi_prime(self, X, y, weights, alpha):
        # calculate the derivative of phi w.r.t alpha using central difference method
        h = 1e-6
        phi_plus = self.phi(X, y, weights, alpha + h)
        phi_minus = self.phi(X, y, weights, alpha - h)
        return (phi_plus - phi_minus) / (2 * h)

    def secant(self, X, y, weights, alpha, alpha_prev):

        phi_prime_curr = self.phi_prime(X, y, weights, alpha)
        phi_prime_prev = self.phi_prime(X, y, weights, alpha_prev)

        new_alpha = alpha - \
            (((alpha - alpha_prev) / (phi_prime_curr -
             phi_prime_prev + 1e-10)) * phi_prime_curr)

        alpha, alpha_prev = new_alpha, alpha

        # if abs(self.phi(X, y, weights, new_alpha) - phi_prime_curr) < 1e-4:
        if abs(new_alpha - alpha_prev) < 1e-8:
            return new_alpha
        else:
            return self.secant(X, y, weights, alpha, alpha_prev)

    def optimize_alpha(self, X, y, weights):
        alpha = 0.01
        alpha_prev = 0.1
        new_alpha = self.secant(X, y, weights, alpha, alpha_prev)
        if new_alpha < 1e-4:
            new_alpha = 1e-4
        elif new_alpha > 0.9:
            new_alpha = 0.9
        return new_alpha

    def fit(self, X, y, alpha_callback=None):
        X = self._preprocess_input_data(X)
        self.weights = self._weights(X.shape[1])
        X = self._bias(X)

        for i in range(self.max_iter):

            weight = self.weights

            alpha = self.optimize_alpha(X, y, weight)

            if alpha_callback is not None:
                alpha_callback(alpha)

            y_hat = self._y_hat(X, self.weights)
            self.gradient = self._cal_gradient(y_hat, y, X)

            self.weights = self.weights - alpha * self.gradient

            # keeping track of cost/loss
            y_hat = self._y_hat(X, self.weights)
            self.loss_history.append(self._cal_loss(y_hat, y))

            # Break the loop if loss is not changing much
            if i > 0 and self.tol is not None:
                if abs(self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2] < self.tol:
                    print(
                        f"Loss is not changing much at iteration {i}. Hence, stopping the training")
                    break

            # break the loop if loss is nan
            if np.isnan(self.loss_history[-1]):
                print(
                    f"Loss is nan at iteration {i}. Hence, stopping the training")
                break


class LogisticGD(BaseGD):
    def __init__(self, alpha, optimizer: list["BatchGD", "MiniBatchGD", "StochasticGD"], bias=False, tol=None, max_iter=100, **kwargs):
        """
        Parameters
        ----------
        alpha : float
            Learning rate

        optimizer : str
            Optimizer to use for training

        bias : bool, default False
            Whether to include bias term or not

        tol : float, default None
            Tolerance for stopping criteria

        max_iter : int, default 100
            Maximum number of iterations

        batch_size : int, default 32
            Only if choosen optimizer MiniBatch
        """

        self.alpha = alpha
        self.optimizer = optimizer
        self.bias = bias
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer_dict = {
            "BatchGD": BatchGD, "StochasticGD": StochasticGD, "MiniBatchGD": MiniBatchGD}
        self.weights = None
        self.loss_history = []

        # batch_size only for MiniBatchGD
        self.batch_size = kwargs.get("batch_size", 32)

    def accuracy(self, y_true, y_pred):
        """
        Parameters
        ----------
        y_true : numpy.ndarray , shape (m_samples, )
            Target values
        y_pred : numpy.ndarray , shape (m_samples, )
            Predicted values

        Returns
        -------
        accuracy : float
            Accuracy

        """
        return np.sum(y_true == y_pred)/len(y_true)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray , shape (m_samples, n_features)
            Training data
        y : numpy.ndarray , shape (m_samples, )
            Target values

        Returns
        -------
            Nothing
        """

        # check if optimizer is valid
        optimizer_class = self.optimizer_dict.get(self.optimizer, None)
        if optimizer_class is None:
            raise ValueError("Invalid optimizer")

        # initialize the optimizer
        optimizer = optimizer_class(
            alpha=self.alpha, max_iter=self.max_iter, bias=self.bias, tol=self.tol)

        # if optimizer is MiniBatchGD then set batch_size
        if self.optimizer == "MiniBatchGD":
            optimizer.batch_size = self.batch_size

        # set the loss function and y_hat function in optimizer
        optimizer._cal_loss = self._cal_loss
        optimizer._y_hat = self._y_hat

        # fit the model
        optimizer.fit(X, y)

        # set the weights and loss history to the class's attributes
        self.weights = optimizer.weights
        self.loss_history = optimizer.loss_history

    def prob_predict(self, X):
        X = self._preprocess_input_data(X)
        X = self._bias(X)
        return self._sigmoid(super().predict(X))

    def predict(self, X):
        return np.where(self.prob_predict(X) >= 0.5, 1, 0)

    def _y_hat(self, X, w):
        return self._sigmoid(np.dot(X, w.T))

    def _cal_loss(self, y_hat, y_true):
        """
        Parameters
        ----------
        y_hat : numpy.ndarray , shape (m_samples, )
            Predicted values
        y_true : numpy.ndarray , shape (m_samples, )
            Target values

        Returns
        -------
        cost : float
            Cost

        """
        # no. of samples
        m = len(y_true)

        # calculate cost in term of y_true and y_hat
        total_cost = np.sum(-y_true * np.log(y_hat) -
                      (1 - y_true) * (np.log(1 - y_hat)))

        if self.penalty is None:
            return total_cost
        elif self.penalty == "l1":
            regularization = (self.lambda_ / m) * np.sum(np.abs(self.weights))
            return total_cost + regularization
        elif self.penalty == "l2":
            regularization = (self.lambda_ / (2*m)) * np.sum(self.weights**2)



    def _sigmoid(self, z):
        # code for calculating sigmoid
        return 1 / (1 + np.exp(-z))
