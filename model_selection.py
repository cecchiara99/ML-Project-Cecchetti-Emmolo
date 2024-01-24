import numpy as np
import copy as cp
from chiara_prova import NeuralNetwork
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split

from itertools import product

import numpy as np
from itertools import product

import numpy as np
from itertools import product

import numpy as np
from itertools import product

import numpy as np
from itertools import product

def generate_combinations_from_ranges(iperparametri_ranges):
    hyperparameters = []

    for key, (lower_limit, upper_limit, step) in iperparametri_ranges.items():
        values = np.arange(lower_limit, upper_limit + 0.0001, step)  # Adding a small value to include upper_limit
        hyperparameters.append({key: values})

    all_combinations = []
    for params in hyperparameters:
        key = list(params.keys())[0]
        values = params[key]
        all_combinations.append((key, values))

    combinazioni_valori = list(product(*[params[1] for params in all_combinations]))

    result_combinations = []
    for combinazione in combinazioni_valori:
        dizionario_combinazione = {param[0]: round(combinazione[i], 3) if param[0] == 'lambda_reg' else round(combinazione[i], 2) for i, param in enumerate(all_combinations)}
        result_combinations.append(dizionario_combinazione)
        print(dizionario_combinazione)

    return result_combinations





def k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparams, K):
    network = None
    best_theta = None
    best_model = None
    best_validation_error = float('inf')

    # Cycle for grid search
    for theta in hyperparams:
        tot_validation_error = 0.0
        hidden_size = theta['hidden_size']
        theta.pop('hidden_size')
        print(f"\nCurrent hyperparameters: {theta}\n")
        # Cycle for K-fold cross validation
        for k in range(K):
            # Split the data into training and validation sets
            training_X, training_y, validation_X, validation_y = split_data_into_folds(data_X, data_y, K, k)
            
            # Train the model on the training set
            network = NeuralNetwork(input_size, hidden_size, output_size, activation_hidden, activation_output, **theta)
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

        theta['hidden_size'] = hidden_size

        # Compute the average validation error
        avg_validation_error = tot_validation_error / K

        print(f"\nAverage validation error: {avg_validation_error}\n")

        # Update best hyperparameter and best model if the current ones are better
        if avg_validation_error < best_validation_error:
            best_validation_error = avg_validation_error
            best_theta = cp.deepcopy(theta)
            best_model = cp.deepcopy(network)

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


import matplotlib.pyplot as plt

def model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, K):
    """
    Select the final model using K-fold cross validation

    :param data_X: the input data
    :param data_y: the target data
    :param K: number of folds

    :return: the final model
    """

    iperparametri =  {
    'hidden_size': (2, 5, 1),      # Specify range (lower_limit, upper_limit, step)
    'learning_rate': (0.1, 0.9, 0.01),  # Specify range (lower_limit, upper_limit, step)
    'epochs': (200, 1000, 200),        # Specify range (lower_limit, upper_limit, step)
    'batch_size': (128, 128, 64),      # Specify range (lower_limit, upper_limit, step)
    'momentum': (0.5, 0.9, 0.01),        # Specify range (lower_limit, upper_limit, step)
    'lambda_reg': (0.001, 0.1, 0.001),  # Specify range (lower_limit, upper_limit, step)
}

    hyperparameters = generate_combinations_from_ranges(iperparametri)

    

    # Select the best hyperparameters and best model using K-fold cross validation
    best_theta, best_model = k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, K)

    print(f"Best hyperparameters: {best_theta}")
    X_train, X_val, y_train, y_val = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
    print(f"Best validation error (Pre-Training): {best_model.evaluate(X_val, y_val)}")

    # Train the model on the whole training set using the best hyperparameters
    best_model.train(data_X, data_y)
    
    X_train, X_val, y_train, y_val = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
    # Evaluate on validation set
    print(f"Best validation error (Post-Training): {best_model.evaluate(X_val, y_val)}")
    

    # Compute accuracy on validation set
    val_predictions = best_model.predict(X_val)
    accuracy = best_model.compute_accuracy(y_val, val_predictions)
    

    print(" Accuracy: ", accuracy)

    

    # Copy the best model
    final_model = cp.deepcopy(best_model)

    return final_model


# DA SPOSTARE
def model_assessment(final_model, test_data):
    # Return the test error
    pass


def predict(self, X):
        _, output = self.forward_propagation(X)
        return np.round(output)

def compute_accuracy(self, y_true, y_pred):
    return np.mean(y_true == y_pred)