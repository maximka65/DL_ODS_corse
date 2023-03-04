import numpy as np


def formula_soft(x):
    return np.exp(x - np.max(x)) / (np.sum(np.exp(x - np.max(x))))

def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        
        return formula_soft(predictions)
    else:
        return np.apply_along_axis(formula_soft, 1, predictions)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    
    true = np.zeros(probs.shape)
    if type(target_index) == type(1):
        true[target_index] = 1
    else:
        count = 0
        for sample_true in target_index:
            true[count][sample_true] = 1
            count += 1

    return - np.sum(true * np.log(probs))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    y_true = np.zeros(predictions.shape)
    if isinstance(target_index, int):
        y_true[target_index] = 1
    else:
        count = 0
        for sample_y_true in target_index:
            y_true[count][sample_y_true] = 1
            count += 1

    softmax_prob = softmax(predictions)
    loss = cross_entropy_loss(softmax_prob, target_index)

    dprediction = softmax_prob - y_true

    return loss, dprediction


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



def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    loss, dW = softmax_with_cross_entropy(predictions, target_index)

    dW = X.transpose().dot(dW)

    return loss, dW

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_indices in batches_indices:
                # Compute loss and gradients
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                loss, dW = linear_softmax(batch_X, self.W, batch_y)

                # Apply gradient to weights using learning rate
                loss_reg, dW_reg = l2_regularization(self.W, reg)
                loss += loss_reg
                dW += dW_reg
                self.W -= learning_rate * dW

                loss_history.append(loss)
        return loss_history


    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        '''
        scores = np.dot(X, self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

