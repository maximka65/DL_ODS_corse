import numpy as np

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(np.square(W))

    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions: np.ndarray, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    and their gradients w.r.t. loss

    Arguments:
    predictions, np array, shape (batch_size, num_classes) - model predictions
    target_index, np array of int, shape (batch_size) - index of true classes for each example

    Returns:
    loss, single value - cross-entropy loss
    dprediction, np array same shape as predictions - gradient of predictions by loss value
    """

    # TODO: Implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    probs = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)

    batch_size = predictions.shape[0]

    loss = -np.sum(np.log(probs[np.arange(batch_size), target_index])) / batch_size
    dprediction = probs.copy()
    dprediction[np.arange(batch_size), target_index] -= 1
    dprediction /= batch_size

    return loss, dprediction



class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, d_out):
        d_result = np.multiply(d_out, self.X > 0)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        d_input = np.dot(d_out, self.W.value.T)
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0, keepdims=True)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
