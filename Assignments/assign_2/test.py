import numpy as np
import pytest
from GradientDescent import *

# Create a test dataset
X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_test = np.array([1, 2, 3])


def test_BaseGD_init():
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    assert base_gd.alpha == 0.01
    assert base_gd.max_iter == 1000
    assert base_gd.tol is None
    assert base_gd.bias == True
    assert base_gd.weights is None


def test_BaseGD_bias():
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    X_bias = base_gd._bias(X_test)
    assert X_bias.shape == (3, 4)
    assert X_bias[0, 0] == 1
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=False, tol=0.0001)
    X_nobias = base_gd._bias(X_test)
    assert np.array_equal(X_nobias, X_test) #X_nobias is X_test, b/c X_nobias is float


def test_BaseGD_weights():
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    weights = base_gd._weights(3)
    assert weights.shape == (4,)
    assert isinstance(weights[0], float)
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=False, tol=0.0001)
    weights = base_gd._weights(3)
    assert weights.shape == (3,)
    assert isinstance(weights[0], float)


def test_BaseGD_y_hat():
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    weights = np.array([1, 2, 3, 4])
    X_bias = base_gd._bias(X_test)
    y_hat = base_gd._y_hat(X_bias, weights)
    assert y_hat.shape == (3,)
    assert y_hat[0] == 21


def test_BaseGD_cal_loss():
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    weights = np.array([1, 2, 3, 4])
    X_bias = base_gd._bias(X_test)
    y_hat = base_gd._y_hat(X_bias, weights)
    cost = base_gd._cal_loss(y_hat, y_test)
    assert isinstance(cost, float)
    assert cost > 0


def test_BaseGD_cal_gradient():
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    weights = np.array([1, 2, 3, 4])
    X_bias = base_gd._bias(X_test)
    y_hat = base_gd._y_hat(X_bias, weights)
    gradient = base_gd._cal_gradient(y_hat, y_test,X_bias)
    assert gradient.shape == (4,)
    assert isinstance(gradient[0], float)


def test_BaseGD_predict_no_weights():
    base_gd = BaseGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    X_bias = base_gd._bias(X_test)
    with pytest.raises(AttributeError) as e:
        base_gd.predict(X_bias)
    assert str(e.value) == "You have to fit the model first"


###################################################################
def test_BatchGD_predict():
    batch_gd = BatchGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    batch_gd.fit(X_test, y_test)
    y_pred = batch_gd.predict(X_test)
    assert y_pred.shape == (3,)
    assert isinstance(y_pred[0], float)


def test_BatchGD_predict_no_weights():
    batch_gd = BatchGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    with pytest.raises(AttributeError):
        batch_gd.predict(X_test)


def test_BatchGD_predict_no_bias():
    batch_gd = BatchGD(alpha=0.01, max_iter=1000, bias=False, tol=0.0001)
    batch_gd.fit(X_test, y_test)
    X_nobias = batch_gd._bias(X_test)
    y_pred = batch_gd.predict(X_nobias)
    assert y_pred.shape == (3,)
    assert isinstance(y_pred[0], float)


def test_BatchGD_predict_after_fitting_with_bias_false():
    batch_gd = BatchGD(alpha=0.01, max_iter=1000, bias=False, tol=0.0001)
    batch_gd.fit(X_test, y_test)
    X_nobias = batch_gd._bias(X_test)
    y_pred = batch_gd.predict(X_nobias)
    assert y_pred.shape == (3,)
    assert isinstance(y_pred[0], float)


def test_BatchGD_predict_after_fitting_with_bias_true():
    batch_gd = BatchGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    batch_gd.fit(X_test, y_test)
    # X_bias = batch_gd._bias(X_test)
    y_pred = batch_gd.predict(X_test)
    assert batch_gd.weights.shape == (4,)
    assert y_pred.shape == (3,)
    assert isinstance(batch_gd.weights[0], float)
    assert isinstance(y_pred[0], float)


###################################################################

def test_MiniBatchGD_init():
    mini_batch_gd = MiniBatchGD(
        alpha=0.01, max_iter=1000, bias=True, tol=0.0001, batch_size=32)
    assert mini_batch_gd.alpha == 0.01
    assert mini_batch_gd.max_iter == 1000
    assert mini_batch_gd.tol is None
    assert mini_batch_gd.bias == True
    assert mini_batch_gd.batch_size == 32
    assert mini_batch_gd.weights is None


def test_MiniBatchGD_fit():
    mini_batch_gd = MiniBatchGD(
        alpha=0.01, max_iter=1000, bias=True, tol=0.0001, batch_size=32)
    mini_batch_gd.fit(X_test, y_test)
    assert mini_batch_gd.weights.shape == (4,)
    assert isinstance(mini_batch_gd.weights[0], float)


def test_MiniBatchGD_predict():
    mini_batch_gd = MiniBatchGD(
        alpha=0.01, max_iter=1000, bias=True, tol=0.0001, batch_size=32)
    mini_batch_gd.fit(X_test, y_test)
    y_pred = mini_batch_gd.predict(X_test)
    assert y_pred.shape == (3,)
    assert isinstance(y_pred[0], float)


def test_MiniBatchGD_predict_no_weights():
    mini_batch_gd = MiniBatchGD(
        alpha=0.01, max_iter=1000, bias=True, tol=0.0001, batch_size=32)
    with pytest.raises(AttributeError):
        mini_batch_gd.predict(X_test)

###########################################################


def test_StochasticGD_init():
    sgd = StochasticGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    assert sgd.alpha == 0.01
    assert sgd.max_iter == 1000
    assert sgd.tol is None
    assert sgd.bias == True
    assert sgd.weights is None


def test_StochasticGD_bias():
    sgd = StochasticGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    X_bias = sgd._bias(X_test)
    assert X_bias.shape == (3, 4)
    assert X_bias[0, 0] == 1
    sgd = StochasticGD(alpha=0.01, max_iter=1000, bias=False, tol=0.0001)
    X_nobias = sgd._bias(X_test)
    assert np.array_equal(X_nobias, X_test) #X_nobias is X_test, b/c X_nobias is float


def test_StochasticGD_weights():
    sgd = StochasticGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    weights = sgd._weights(3)
    assert weights.shape == (4,)
    assert isinstance(weights[0], float)
    sgd = StochasticGD(alpha=0.01, max_iter=1000, bias=False, tol=0.0001)
    weights = sgd._weights(3)
    assert weights.shape == (3,)
    assert isinstance(weights[0], float)


def test_StochasticGD_y_hat():
    sgd = StochasticGD(alpha=0.01, max_iter=1000, bias=True, tol=0.0001)
    weights = np.array([1, 2, 3, 4])
    X_bias = sgd._bias(X_test)
    y_hat = sgd._y_hat(X_bias, weights)
    assert y_hat.shape == (3,)
