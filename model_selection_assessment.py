import numpy as np
import copy as cp
from neural_network import NeuralNetwork
from itertools import product
from utils import *
import json
import shutil
import os

def model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, test_X, test_y, task, type_selection = "k-fold"):
    """
    Perform model selection to select the best results

    :param input_size: the size of the input layer
    :param output_size: the size of the output layer
    :param activation_hidden: the activation function of the hidden layer
    :param activation_output: the activation function of the output layer
    :param data_X: the input data
    :param data_y: the target data
    :param test_X: the test input data
    :param test_y: the test target data
    :param task: the task to perform
    :param type_selection: the type of model selection to perform

    :return: the best results of the model selection
    """

    test_losses = []
    best_results = None
    patience = 20

    #hyperparameters_ranges = set_hyperparameters_ranges(task, len_data=data_X.shape[0], fine_search=True)

    #hyperparameters = generate_combinations_from_ranges(hyperparameters_ranges)

    # Monk1
    #hyperparameters = [{'hidden_size': 3, 'learning_rate': 0.05, 'epochs': 500, 'batch_size': 1, 'momentum': 0.9, 'lambda_reg': 0.001, 'w_init_limit': [-0.2, 0.2]}]
    # Monk2
    #hyperparameters = [{'hidden_size': 4, 'learning_rate': 0.4, 'epochs': 500, 'batch_size': 1, 'momentum': 0.9, 'lambda_reg': 0, 'w_init_limit': [-0.2, 0.2]}]
    # Monk3 no reg
    #hyperparameters = [{'hidden_size': 2, 'learning_rate': 0.55, 'epochs': 500, 'batch_size': 1, 'momentum': 0.9, 'lambda_reg': 0, 'w_init_limit': [-0.2, 0.2]}]
    # Monk3 reg
    #hyperparameters = [{'hidden_size': 3, 'learning_rate': 0.1, 'epochs': 500, 'batch_size': 1, 'momentum': 0.9, 'lambda_reg': 0.001, 'w_init_limit': [-0.2, 0.2]}]
    # Cup
    #hyperparameters = [{'hidden_size': 47, 'learning_rate': 0.09, 'epochs': 600, 'batch_size': 200, 'momentum': 0.1, 'lambda_reg': 0.00001, 'w_init_limit': [-0.7, 0.7]}]
    
    #hyperparameters = [{'hidden_size': 4, 'learning_rate': 0.1, 'epochs': 500, 'batch_size': len(data_y), 'momentum': 0.9, 'lambda_reg': 0.1, 'w_init_limit': [-0.7, 0.7]}]
    #hyperparameters = [{'hidden_size': 30, 'learning_rate': 0.2, 'epochs': 600, 'batch_size': 200, 'momentum': 0.1, 'lambda_reg': 0.001, 'w_init_limit': [-0.1, 0.1]}]
    #hyperparameters = [{'hidden_size': 30, 'learning_rate': 0.1, 'epochs': 600, 'batch_size': 16, 'momentum': 0.1, 'lambda_reg': 0.0001, 'w_init_limit': [-0.5, 0.5]}]
    hyperparameters = [{'hidden_size': 5, 'learning_rate': 0.05, 'epochs': 1000, 'batch_size': 200, 'momentum': 0.5, 'lambda_reg': 0.0001, 'w_init_limit': [-0.1, 0.1]}]

    # MODEL SELECTION
    if type_selection == "k-fold":
        K = 5
        print(f"\nINIZIO K-fold cross validation, K = {K} \n")

        if task != "cup":
            # For MONK
            best_results = k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, K, task, patience, n_best_results=1)
        else:
            # For CUP
            best_results = k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, K, task, patience, n_best_results=1)

    elif type_selection == "hold-out":
        print("\nINIZIO Hold-out\n")

        if task != "cup":
            # For MONK
            best_results = hold_out(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, task, patience, n_best_results=1)
        else:
            # For CUP
            best_results = hold_out(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, task, patience, n_best_results=1)
    else:
        print("Error: type of model selection not recognized")
        return None
    
    i = 0 # Counter for the models
    # RETRAINING
    print("END MODEL SELECTION, START RETRAINING...\n")
    if task != "cup":
        # For MONK
        for best_result in best_results:
            i += 1
            print(f"\nBest theta: {best_result['theta']}\n")
            print(f"\nBest validation error: {best_result['validation_error']}\n")

            # Retrain the model on the whole training set
            model = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **best_result['theta'])
            losses, test_losses, accuracies, test_accuracies, epochs = model.retrain(data_X, data_y, test_X, test_y, task, patience)
            best_result['model'] = cp.deepcopy(model)

            # Compute accuracy on development set and test set - after retraining
            accuracy = model.compute_accuracy(data_y, model.predict(data_X))
            print(f"ACCURACY after retraining: {accuracy}")
            accuracy_test = model.compute_accuracy(test_y, model.predict(test_X))
            print(f"ACCURACY TEST after retraining: {accuracy_test}")
            
            # Plot the graphs and save them
            plot_graphs_monk(losses, test_losses, accuracies, test_accuracies, epochs)
            # Copy the image file to the destination folder for the learning curve
            source_file_learning = './learning_curve.png'
            destination_file_learning = './models_graphs/' + task + '_model' + str(i) + '_' + type_selection + '_learning_curve.jpg'
            with open(source_file_learning, 'rb') as f:
                with open(destination_file_learning, 'wb+') as f1:
                    shutil.copyfileobj(f, f1)
            # Copy the image file to the destination folder for the accuracy graph
            source_file_accuracy = './accuracy_curve.png'
            destination_file_accuracy = './models_graphs/' + task + '_model' + str(i) + '_'+ type_selection  + '_accuracy.jpg'
            with open(source_file_accuracy, 'rb') as f:
                with open(destination_file_accuracy, 'wb+') as f1:
                    shutil.copyfileobj(f, f1)
            
    else:
        # For CUP
        for best_result in best_results:
            i += 1
            print(f"\nBest theta: {best_result['theta']}\n")
            print(f"\nBest validation error: {best_result['validation_error']}\n")

            # Retrain the model on the whole training set
            model = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **best_result['theta'])
            mees, mees_test, _, _, epochs = model.retrain(data_X, data_y, test_X, test_y, task, patience)
            best_result['model'] = cp.deepcopy(model)

            # Compute MEE on development set and test set - after retraining
            mee_after_retraining = model.evaluate(data_X, data_y, task)
            print(f"MEE after retraining: {mee_after_retraining}")
            mee_test_after_retraining = model.evaluate(test_X, test_y, task)
            print(f"MEE TEST after retraining: {mee_test_after_retraining}")

            # Save the best models info in a json file for CUP    
            if task == "cup":
                best_model_info = {
                    'model': {
                        'hidden_size': model.hidden_size,
                        'learning_rate': model.learning_rate,
                        'epochs': model.epochs,
                        'batch_size': model.batch_size,
                        'momentum': model.momentum,
                        'lambda_reg': model.lambda_reg,
                        'w_init_limit': model.w_init_limit
                    },
                    'mee': mee_after_retraining,
                    'mee_test': mee_test_after_retraining,
                    'validation_error': best_result['validation_error']
                }
                destination_folder = './models_info'
                # Check if the folder exists, if not, create it
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                # Save the model info in a json file
                with open('./models_info/'+ task + '_model' + str(i) + '_' + type_selection + '_model_info.json', 'w+') as outfile:
                    json.dump(best_model_info, outfile)

            # Plot the graphs and save them
            plot_graphs_cup(mees, mees_test, epochs)
            # Copy the image file to the destination folder for the MEE curve
            source_file_mee = './mee_curve.png'
            destination_file_mee = './models_graphs/' + task + '_model' + str(i) + '_' + type_selection + '_mee.jpg'
            with open(source_file_mee, 'rb') as f:
                with open(destination_file_mee, 'wb+') as f1:
                    shutil.copyfileobj(f, f1)

    return best_results
  
def k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparams, K, task, patience, n_best_results):
    """
    Perform K-fold cross validation to select the best results

    :param input_size: the size of the input layer
    :param output_size: the size of the output layer
    :param activation_hidden: the activation function of the hidden layer
    :param activation_output: the activation function of the output layer
    :param data_X: the input data
    :param data_y: the target data
    :param hyperparams: a list of dictionaries containing the hyperparameters values to try
    :param K: number of folds
    :param task: the task to perform
    :param patience: the number to wait before early stopping
    :param n_best_results: the number of best results to return

    :return: the best results of the model selection
    """

    network = None
    best_results = initialize_best_results(n_best_results)
    left_combinations = len(hyperparams)

    # Cycle for grid search
    for theta in hyperparams:
        tot_validation_error = 0.0
  
        # 
        indices = np.random.permutation(len(data_y))
        X_shuffled = data_X[indices]
        y_shuffled = data_y[indices]

        # Cycle for K-fold cross validation
        for k in range(K):
            # Split the data into training and validation sets
            training_X, training_y, validation_X, validation_y = split_data_into_folds(X_shuffled, y_shuffled, K, k)
            
            # Train the model on the training set, and evaluate it on the validation set
            network = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **theta)
            network.train(training_X, training_y, validation_X, validation_y, task, patience)
            validation_error = network.evaluate(validation_X, validation_y, task)
            
            tot_validation_error += validation_error
        
        # Compute the average validation error
        avg_validation_error = tot_validation_error / K

        if avg_validation_error < best_results[-1]['validation_error']:
            best_results[-1]['theta'] = cp.deepcopy(theta)
            best_results[-1]['model'] = cp.deepcopy(network)
            best_results[-1]['validation_error'] = avg_validation_error
            best_results.sort(key=lambda x: x['validation_error'])
            print(f"\nValidation error: {avg_validation_error}\n")

        left_combinations -= 1
        print(f"\nCombinations left: {left_combinations}\n")
            
    return best_results
        
    
def hold_out(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameter, task, patience, n_best_results):
    """
    Perform hold-out to select the best results

    :param input_size: the size of the input layer
    :param output_size: the size of the output layer
    :param activation_hidden: the activation function of the hidden layer
    :param activation_output: the activation function of the output layer
    :param data_X: the input data
    :param data_y: the target data
    :param hyperparameter: a list of dictionaries containing the hyperparameters values to try
    :param task: the task to perform
    :param patience: the number to wait before early stopping
    :param n_best_results: the number of best results to return

    :return: the best results of the model selection
    """

    network = None
    best_results = initialize_best_results(n_best_results)
    left_combinations = len(hyperparameter)
    
    for theta in hyperparameter:
        indices = np.random.permutation(len(data_y))
        X_shuffled = data_X[indices]
        y_shuffled = data_y[indices]

        training_X, training_y, validation_X, validation_y = split_data(X_shuffled, y_shuffled, 0.8)

        network = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **theta)
        network.train(training_X, training_y, validation_X, validation_y, task, patience)
        validation_error = network.evaluate(validation_X, validation_y, task)
        

        if validation_error < best_results[-1]['validation_error']:
            best_results[-1]['theta'] = cp.deepcopy(theta)
            best_results[-1]['model'] = cp.deepcopy(network)
            best_results[-1]['validation_error'] = validation_error
            best_results.sort(key=lambda x: x['validation_error'])
            print(f"\nValidation error: {validation_error}\n")

        left_combinations -= 1
        print(f"Combinations left: {left_combinations}\n")

    return best_results


def generate_combinations_from_ranges(hyperparameters_ranges):
    """
    Generate all the possible combinations of hyperparameters values from the specified ranges

    :param hyperparameters_ranges: a dictionary containing the hyperparameters names as keys and the ranges as values

    :return: a list of dictionaries containing the hyperparameters values combinations
    """

    hyperparameters = []

    for key, value in hyperparameters_ranges.items():
        if key == 'w_init_limit' and isinstance(value, list):
            # Flatten the list of pairs
            values = [item for sublist in value for item in sublist]
        elif (key == 'lambda_reg' or key == 'batch_size' ) and isinstance(value, list):
            # Use the specified list
            values = value
        else:
            # Use the specified range
            lower_limit, upper_limit, step = value
            values = np.arange(lower_limit, upper_limit + 0.0001, step)

        if key == 'w_init_limit':
            # Create sliding window pairs
            values = [[values[i], values[i+1]] for i in range(0, len(values), 2)]

        hyperparameters.append({key: values})

    all_combinations = []
    for params in hyperparameters:
        key = list(params.keys())[0]
        values = params[key]
        all_combinations.append((key, values))

    values_combinations = list(product(*[params[1] for params in all_combinations]))

    result_combinations = []

    for combination in values_combinations:
        dictionary_combination = {
            param[0]: round(combination[i], 2)
            for i, param in enumerate(all_combinations) if param[0] != 'w_init_limit'
        }
        # Add the 'w_init_limit' and 'lambda_reg' keys without rounding
        w_init_limit_index = [i for i, param in enumerate(all_combinations) if param[0] == 'w_init_limit'][0]
        lambda_reg_index = [i for i, param in enumerate(all_combinations) if param[0] == 'lambda_reg'][0]
        dictionary_combination['w_init_limit'] = combination[w_init_limit_index]
        dictionary_combination['lambda_reg'] = combination[lambda_reg_index]
        result_combinations.append(dictionary_combination)
    
    print("\nHyperparameters combinations generated\n")
    print(f"Number of combinations: {len(result_combinations)}\n")

    return result_combinations