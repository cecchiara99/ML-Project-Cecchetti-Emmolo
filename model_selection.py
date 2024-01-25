import numpy as np
import copy as cp
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split

def model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, K):
    
    hyperparameters_ranges =  {
        'hidden_size': (2, 4, 2),      # Specify range (lower_limit, upper_limit, step)
        'learning_rate': (0.1, 0.9, 0.8),  # Specify range (lower_limit, upper_limit, step)
        'epochs': (200, 1000, 800),        # Specify range (lower_limit, upper_limit, step)
        'batch_size': (128, 128, 64),      # Specify range (lower_limit, upper_limit, step)
        'momentum': (0.5, 0.9, 0.4),        # Specify range (lower_limit, upper_limit, step)
        'lambda_reg': (0.001, 0.1, 0.001),  # Specify range (lower_limit, upper_limit, step)
    }

    hyperparameters = generate_combinations_from_ranges(hyperparameters_ranges)

    # Select the best hyperparameters and best model using K-fold cross validation
    best_theta, best_model = k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, K)

    print(f"Best hyperparameters: {best_theta}")
    #X_train, X_val, y_train, y_val = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
    #print(f"Best validation error (Pre-Training): {best_model.evaluate(X_val, y_val)}")

    # Train the model on the whole training set using the best hyperparameters
    best_model.train(data_X, data_y)
    
    #X_train, X_val, y_train, y_val = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
    # Evaluate on validation set
    #print(f"Best validation error (Post-Training): {best_model.evaluate(X_val, y_val)}")
    
    # Compute accuracy on validation set
    val_predictions = best_model.predict(data_X)
    accuracy = best_model.compute_accuracy(data_y, val_predictions)
    print(" Accuracy: ", accuracy)

    

    # Copy the best model
    final_model = cp.deepcopy(best_model)

    return final_model



def k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparams, K):
    network = None
    best_theta = None
    best_model = None
    best_validation_error = float('inf')

    # Cycle for grid search
    for theta in hyperparams:
        tot_validation_error = 0.0
        #print(f"\nCurrent hyperparameters: {theta}\n")
        # Cycle for K-fold cross validation
        for k in range(K):
            # Split the data into training and validation sets
            training_X, training_y, validation_X, validation_y = split_data_into_folds(data_X, data_y, K, k)
            
            # Train the model on the training set
            network = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **theta)
            network.train(training_X, training_y)

            # Evaluate the model on the validation set
            validation_error = network.evaluate(validation_X, validation_y)
            
            tot_validation_error += validation_error

        # Compute the average validation error
        avg_validation_error = tot_validation_error / K
        #print(f"\nAverage validation error: {avg_validation_error}\n")

        # Update best hyperparameter and best model if the current ones are better
        if avg_validation_error < best_validation_error:
            best_validation_error = avg_validation_error
            best_theta = cp.deepcopy(theta)
            best_model = cp.deepcopy(network)
            print(f"\nBest validation error: {best_validation_error}\n")

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



def generate_combinations_from_ranges(hyperparameters_ranges):
    """
    Generate all the possible combinations of hyperparameters values from the specified ranges

    :param hyperparameters_ranges: a dictionary containing the ranges of the hyperparameters

    :return: a list of dictionaries containing all the possible combinations of hyperparameters values
    """

    hyperparameters = []
    
    for key, (lower_limit, upper_limit, step) in hyperparameters_ranges.items():
        values = np.arange(lower_limit, upper_limit + 0.0001, step)  # Adding a small value to include upper_limit
        hyperparameters.append({key: values})

    all_combinations = []
    for params in hyperparameters:
        key = list(params.keys())[0]
        values = params[key]
        all_combinations.append((key, values))

    values_combinations = list(product(*[params[1] for params in all_combinations]))

    result_combinations = []
    for combination in values_combinations:
        dictionary_combination = {param[0]: round(combination[i], 3) if param[0] == 'lambda_reg' else round(combination[i], 2) for i, param in enumerate(all_combinations)}
        result_combinations.append(dictionary_combination)
        #print(dictionary_combination)

    return result_combinations