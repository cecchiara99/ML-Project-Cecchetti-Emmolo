import numpy as np
import copy as cp
from chiara_prova import NeuralNetwork

def k_fold_cross_validation(input_size, hidden_size, data_X, data_y, hyperparams, K):
    network = None
    best_theta = None
    best_model = None
    best_validation_error = float('inf')

    # Cycle for grid search
    for theta in hyperparams:
        tot_validation_error = 0.0

        # Cycle for K-fold cross validation
        for k in range(K):
            # Split the data into training and validation sets
            training_X, training_y, validation_X, validation_y = split_data_into_folds(data_X, data_y, K, k)

            # Train the model on the training set
            network = NeuralNetwork(input_size, hidden_size, **theta) # TODO: aggiungi iperparametri
            network.train(training_X, training_y)

            # Evaluate the model on the validation set
            validation_error = network.evaluate(validation_X, validation_y)
            """
            def evaluate(self, inputs, targets):
                _, predictions = self.forward_propagation(inputs)
                mee = np.mean(np.sqrt((targets - predictions)**2))
                # oppure np.sqrt(np.mean((targets - predictions)**2))
                # oppure np.mean(np.sqrt(np.sum((targets - predictions)**2, axis=1)))
                return mee
            """

            tot_validation_error += validation_error

        # Compute the average validation error
        avg_validation_error = tot_validation_error / K

        # Update best hyperparameter and best model if the current ones are better
        if avg_validation_error < best_validation_error:
            best_validation_error = avg_validation_error
            best_theta = theta
            best_model = cp.copy(network)

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

    # Compute the size of each fold
    n_samples = len(data_X)
    fold_size = n_samples // K

    # Compute the start and end indices of the k-th fold
    start = k * fold_size
    end = (k + 1) * fold_size if k != K - 1 else n_samples # last fold can be bigger

    # Use the k-th fold as validation set
    validation_X = data_X[start:end]
    validation_y = data_y[start:end]

    # Use the rest of the data as training set
    training_X = np.concatenate([data_X[:start], data_X[end:]])
    training_y = np.concatenate([data_y[:start], data_y[end:]])

    return training_X, training_y, validation_X, validation_y


def model_selection(input_size, hidden_size, data_X, data_y, hyperparameters, K):
    """
    Select the final model using K-fold cross validation

    :param data_X: the input data
    :param data_y: the target data
    :param hyperparameters: the hyperparameters to use for grid search
    :param K: number of folds

    :return: the final model
    """

    # Select the best hyperparameters and best model using K-fold cross validation
    best_theta, best_model = k_fold_cross_validation(input_size, hidden_size, data_X, data_y, hyperparameters, K)

    # Train the model on the whole training set using the best hyperparameters
    best_model.train(data_X, data_y)
    # oppure
    #best_model = NeuralNetwork(best_theta)
    #best_model.train(data_X, data_y)
    final_model = cp.copy(best_model)

    return final_model


# DA SPOSTARE
def model_assessment(final_model, test_data):
    # Return the test error
    pass


"""
hyperparams = ...
data = ...
K = ...
test_data = ...

# Train the model on the training set and select the best model
best_model = model_selection(data, hyperparams, K)

# Assess the performance of the best model on the test set
test_error = model_assessment(best_model, test_data)
print(f"Final Test Error: {test_error}")
"""