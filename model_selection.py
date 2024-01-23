import numpy as np
import copy as cp
from NN import NeuralNetwork

def k_fold_cross_validation(network, data_X, data_y, hyperparameters, K):
    best_theta = None
    best_model = None
    best_validation_error = float('inf')

    for theta in hyperparameters:
        tot_validation_error = 0.0

        for k in range(K):
            # Split data into training and validation sets
            D_k_bar_X, D_k_bar_y, D_k_X, D_k_y = split_data_into_folds(data_X, data_y, K, k)

            # Train the model on the counterpart
            network = NeuralNetwork()
            network.train(D_k_bar_X, D_k_bar_y)

            # Estimate risk on the validation part
            validation_error = network.evaluate(D_k_X, D_k_y)

            tot_validation_error += validation_error

            # Reset the model
            network = NeuralNetwork()

        # Global estimation of the risk
        avg_validation_error = tot_validation_error / K

        # Update best hyperparameter if the current one is better
        if avg_validation_error < best_validation_error:
            best_validation_error = avg_validation_error
            best_theta = theta
            #best_model = network.copy_model()

    return best_theta, best_model

def split_data_into_folds(data_X, data_y, K, k):
    """
    Split the data into K folds and return the k-th fold as validation set and the rest as training set

    :param data_X: the input data
    :param data_y: the target data
    :param K: number of folds
    :param k: the k-th fold to use as validation set

    :return: the training and validation sets
    """
    n_samples = len(data_X)
    fold_size = n_samples // K

    start = k * fold_size
    end = (k + 1) * fold_size if k != K - 1 else n_samples # last fold can be bigger

    # Select the k-th fold as validation set
    val_data_X = data_X[start:end]
    val_data_y = data_y[start:end]

    # Use the rest as training set
    train_data_X = np.concatenate([data_X[:start], data_X[end:]])
    train_data_y = np.concatenate([data_y[:start], data_y[end:]])

    return train_data_X, train_data_y, val_data_X, val_data_y

def model_selection(network, data, hyperparameters, K):
    # Select the best hyperparameters using K-fold cross validation
    best_theta, best_model = k_fold_cross_validation(data, hyperparameters, K)

    # Train the model on the whole training set using the best hyperparameters
    network.NeuralNetwork(best_theta)
    final_model = network.train(best_theta, data)

    return final_model # or "return network"

def model_assessment(final_model, test_data):
    # Return the test error
    pass

"""
DA METTE DENTRO NeuralNetwork
def copy_model(self):
    # Crea una nuova istanza del modello
    new_model = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)

    # Copia i parametri del modello corrente nel nuovo modello
    # Questo può variare a seconda della struttura del tuo modello
    new_model.weights_input_hidden = np.copy(self.weights_input_hidden)
    new_model.weights_hidden_output = np.copy(self.weights_hidden_output)
    new_model.bias_hidden = np.copy(self.bias_hidden)
    new_model.bias_output = np.copy(self.bias_output)

    # Altri parametri da copiare, se necessario

    return new_model
"""


hyperparameters = ...
data = ...
K = ...
test_data = ...

# Train the model on the training set and select the best model
best_model = model_selection(data, hyperparameters, K)

# Assess the performance of the best model on the test set
test_error = model_assessment(best_model, test_data)
print(f"Final Test Error: {test_error}")