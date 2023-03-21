from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy

import numpy as np


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input: int, n_output: int, hidden_layer_size: int, reg: float):
        """
        Initializes the neural network

        Arguments:
        n_input: int - dimension of the model input
        n_output: int - number of classes to predict
        hidden_layer_size: int - number of neurons in the hidden layer
        reg: float - L2 regularization strength
        """
        self.reg = reg

        # Create necessary layers
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X: np.ndarray, y: np.ndarray):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X: np array (batch_size, input_features) - input data
        y: np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # Set parameter gradient to zeros
        self.fc1.dW = np.zeros_like(self.fc1.W)
        self.fc1.dB = np.zeros_like(self.fc1.B)
        self.fc2.dW = np.zeros_like(self.fc2.W)
        self.fc2.dB = np.zeros_like(self.fc2.B)

        # Compute loss and fill param gradients by running forward and backward passes through the model
        fc1_out = self.fc1.forward(X)
        relu1_out = self.relu1.forward(fc1_out)
        fc2_out = self.fc2.forward(relu1_out)
        loss, grad = softmax_with_cross_entropy(fc2_out, y)

        # Backpropagate the gradient through the network and accumulate the parameter gradients
        grad = self.fc2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.fc1.backward(grad)

        # Add regularization to the parameter gradients
        self.fc1.dW += self.reg * self.fc1.W
        self.fc2.dW += self.reg * self.fc2.W

        return loss, grad

    def predict(self, X: np.ndarray):
        """
        Produces classifier predictions on the set

        Arguments:
          X: np array (test_samples, num_features)

        Returns:
          y_pred: np.array of int (test_samples)
        """
        # Forward pass
        fc1_out = self.fc1.forward(X)
        relu1_out = self.relu1.forward(fc1_out)
        fc2_out = self.fc2.forward(relu1_out)

        # Output predictions
        pred = np.argmax(fc2_out, axis=1)

        return pred

    def params(self) -> Dict[str, np.ndarray]:
        result = {}

        # Aggregate all of the params
        result['W1'] = self.fc1.W
        result['B1'] = self.fc1.B
        result['W2'] = self.fc2.W
        result['B2'] = self.fc2.B

        return result
