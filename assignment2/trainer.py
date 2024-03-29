from copy import deepcopy

import numpy as np
from metrics import multiclass_accuracy


class Dataset:
    """
    Utility class to hold training and validation data
    """

    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y


class Trainer:
    """
    Trainer of the neural network models
    Perform mini-batch SGD with the specified data, model,
    training parameters and optimization rule
    """

    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-2,
                 learning_rate_decay=1.0):
        """
        Initializes the trainer

        Arguments:
        model - neural network model
        dataset, instance of Dataset class - data to train on
        optim - optimization method (see optim.py)
        num_epochs, int - number of epochs to train
        batch_size, int - batch size
        learning_rate, float - initial learning rate
        learning_rate_decay, float - ratio for decaying learning rate
           every epoch
        """
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.learning_rate_decay = learning_rate_decay

        self.optimizers = None

    def setup_optimizers(self):
        params = self.model.params()
        self.optimizers = {}
        for param_name, param in params.items():
            self.optimizers[param_name] = deepcopy(self.optim)

    def compute_accuracy(self, X, y):
        """
        Computes accuracy on provided data using mini-batches
        """
        indices = np.arange(X.shape[0])
        sections = np.arange(self.batch_size, X.shape[0], self.batch_size)
        batches_indices = np.array_split(indices, sections)

        pred = np.zeros_like(y)

        for batch_indices in batches_indices:
            batch_X = X[batch_indices]
            pred_batch = self.model.predict(batch_X)
            pred[batch_indices] = pred_batch

        return multiclass_accuracy(pred, y)

    def fit(self):
        """
        Trains a model
        """
        if self.optimizers is None:
            self.setup_optimizers()

        num_train = self.dataset.train_X.shape[0]

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(self.num_epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            batch_losses = []

            for batch_indices in batches_indices:
                X_batch = self.dataset.train_X[batch_indices]
                y_batch = self.dataset.train_y[batch_indices]

                # Generate batches and compute loss and gradients for all params
                loss = self.model.compute_loss_and_gradients(X_batch, y_batch)

                # Update params using optimizers
                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    param.value = optimizer.update(param.value, param.grad, self.learning_rate)

                batch_losses.append(loss)

            # Apply learning rate decay
            if np.not_equal(self.learning_rate_decay, 1.0):
                self.learning_rate *= self.learning_rate_decay

            # Compute training and validation accuracy for this epoch
            train_accuracy = self.compute_accuracy(self.dataset.train_X, self.dataset.train_y)
            val_accuracy = self.compute_accuracy(self.dataset.val_X, self.dataset.val_y)

            # Store loss and accuracy history for this epoch
            loss_history.append(np.mean(batch_losses))
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)

            # Print progress
            print("Epoch %i, loss: %f, train accuracy: %f, val accuracy: %f" % (
                epoch + 1, loss_history[-1], train_accuracy, val_accuracy))

        return loss_history, train_acc_history, val_acc_history
