import numpy as np
import copy as cp
from neural_network import NeuralNetwork
from itertools import product
from sklearn.model_selection import train_test_split
from utils import *
import json
import shutil

def model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, K, task, test_X, test_y, type_selection = "k-fold"):

    len_data = data_X.shape[0]
    
    hyperparameters_ranges =  {
        # Specify range (lower_limit, upper_limit, step)
        'hidden_size': (3, 4, 1),           
        'learning_rate': (0.1, 0.9, 0.1),
        'epochs': (300, 600, 100),
        'batch_size': [64, len_data],
        'momentum': (0.5, 0.9, 0.1),
        'lambda_reg': [0.001, 0.01, 0.1],
        'w_init_limit': [[-0.3, 0.3],[-0.2, 0.2],[-0.1,0.1]]
    }

    #hyperparameters = generate_combinations_from_ranges(hyperparameters_ranges)
    #print(f"\nNumber of combinations: {len(hyperparameters)}\n")

    hyperparameters = [{'hidden_size': 3, 'learning_rate': 0.9, 'epochs': 400, 'batch_size': 64, 'momentum': 0.9, 'lambda_reg': 0.001, 'w_init_limit': [-0.3, 0.3]}]

    if type_selection == "k-fold":
        print("\n INIZIO K-fold cross validation\n")
        best_theta, best_model, best_hyperparams, best_validation_error = k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, K, task, test_X, test_y, patience=10)
    elif type_selection == "hold-out":
        print("\n INIZIO Hold-out\n")
        best_theta, best_model, best_hyperparams, best_validation_error = hold_out(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, task, test_X, test_y, patience=10)
    

    print("END MODEL SELECTION, START RETRAINING...\n")
    print(f"Best hyperparameters: {best_theta}")

    if task != "monk3":
        # Train the model on the whole training set using the best hyperparameters
        best_model.train(data_X, data_y, test_X, test_y, task)
    
    if task != "cup":
        # Compute accuracy on validation set
        val_predictions = best_model.predict(data_X)
        accuracy = best_model.compute_accuracy(data_y, val_predictions)
        print("Final accuracy (after retraining): ", accuracy)
    
    # Copy the best model
    final_model = cp.deepcopy(best_model)


    if task == "monk1" or task == "monk2" or task == "monk3":
        # Specify the source and destination file paths
        source_file_learning = './learning_curve.png'
        source_file_accuracy = './accuracy_curve.png'

        destination_file_learning = './model_graphs/' + task+ '_'+ type_selection + '_learning_curve.jpg'
        destination_file_accuracy = './model_graphs/' + task + '_'+ type_selection  + '_accuracy.jpg'

        # Copy and rename the image file
        with open(source_file_learning, 'rb') as f:
            with open(destination_file_learning, 'wb+') as f1:
                shutil.copyfileobj(f, f1)
        
        with open(source_file_accuracy, 'rb') as f:
            with open(destination_file_accuracy, 'wb+') as f1:
                shutil.copyfileobj(f, f1)

        model_info_monk = {
            'theta': best_theta,
            'model_selection': type_selection,
            'validation_error': best_validation_error,
            'accuracy': accuracy,
            'img_learning_curve': destination_file_learning,
            'img_accuracy': destination_file_accuracy,
        }

        # Save the model info in a json file
        with open('./models_info/'+ task+ '_' + type_selection +'_model_info.json', 'w+') as outfile:
            json.dump(model_info_monk, outfile)

    return final_model
  
def k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparams, K, task, test_X, test_y, patience):
    """
    Perform K-fold cross validation to select the best hyperparameters and the best model

    :param input_size: the size of the input layer
    :param output_size: the size of the output layer
    :param activation_hidden: the activation function of the hidden layer
    :param activation_output: the activation function of the output layer
    :param data_X: the input data
    :param data_y: the target data
    :param hyperparams: a list of dictionaries containing the hyperparameters values to try
    :param K: number of folds

    :return: the best hyperparameters and the best model
    """

    network = None
    best_theta = None
    best_model = None
    best_validation_error = float('inf')
    best_accuracy = 0.0
    accuracies = []

    best_hyperparams = []

    left_combinations = len(hyperparams)

    count_patience = 0

    # Cycle for grid search
    for theta in hyperparams:
        tot_validation_error = 0.0
        #print(f"\nCurrent hyperparameters: {theta}\n")
        K = 5
        indices = np.random.permutation(len(data_y))
        X_shuffled = data_X[indices]
        y_shuffled = data_y[indices]

        # Cycle for K-fold cross validation
        for k in range(K):
            # Split the data into training and validation sets
            training_X, training_y, validation_X, validation_y = split_data_into_folds(X_shuffled, y_shuffled, K, k)
            
            # Train the model on the training set
            network = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **theta)
            network.train(training_X, training_y, test_X, test_y, task)
            
            # Evaluate the model on the validation set
            validation_error = network.evaluate(validation_X, validation_y, task)
            val_predictions = network.predict(validation_X)
            accuracy = network.compute_accuracy(validation_y, val_predictions)
            accuracies.append(accuracy)
            
            tot_validation_error += validation_error


        # Compute the average validation error
        avg_validation_error = tot_validation_error / K
        #print(f"\nAverage validation error: {avg_validation_error}\n")

        # Update best hyperparameter and best model if the current ones are better
        if avg_validation_error < best_validation_error:
            count_patience = 0
            best_accuracy = accuracies[-1]
                
            best_validation_error = avg_validation_error
            best_theta = cp.deepcopy(theta)
            best_model = cp.deepcopy(network)
            
            if task == "cup":
                model = {
                    'theta': best_theta,
                    'validation_error': best_validation_error,
                    # AGGIUNGERE GRAFICI
                }
                if len(best_hyperparams) >= 5:
                    best_hyperparams.pop(0)
                    best_hyperparams.append(model)
            
            print(f"\nNEW BETTER!!\nBest validation error: {best_validation_error}\nBest hyperparameters: {theta}\nBest accuracy: {best_accuracy}\n")
        else:
            count_patience += 1
            if count_patience == patience:
                print("Early stopping after ", patience, " iterations")
                break
        
        left_combinations -= 1
        print(f"\nCombinations left: {left_combinations}\n")
    
    return best_theta, best_model, best_hyperparams, best_validation_error

def hold_out(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameter, task, test_X, test_y, patience):
        
        val_losses, accuracies = [], []

        network = None
        best_theta = None
        best_model = None
        left_combinations = len(hyperparameter)
        best_hyperparams = []
        count_patience = 0
        
        for theta in hyperparameter:
            X_train, y_train, X_val, y_val = split_data(data_X, data_y, 0.9)

            indices = np.random.permutation(len(y_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            network = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **theta)
            network.train(X_shuffled, y_shuffled, test_X, test_y, task)

            # Evaluate on validation set
            val_loss = network.evaluate(X_val, y_val, task)
            val_losses.append(val_loss)

            # Compute accuracy on validation set
            val_predictions = network.predict(X_val)
            accuracy = network.compute_accuracy(y_val, val_predictions)
            accuracies.append(accuracy)

            # Update best hyperparameter and best model if the current ones are better
            if len(val_losses) > 1:
                if val_loss < val_losses[-2]:
                    best_theta = cp.deepcopy(theta)
                    best_model = cp.deepcopy(network)
                    count_patience = 0
                    
                    print(f"\nBest validation error: {val_loss}\nBest hyperparameters: {theta}\nBest accuracy: {accuracy}\n")
                
                else:
                    count_patience += 1
                    print(f"\nCurrent validation error: {val_loss}\nCurrent hyperparameters: {theta}\nCurrent accuracy: {accuracy}\n")
                    if count_patience == patience:
                        print("Early stopping after ", patience, " iterations")
                        break

            else:
                best_theta = cp.deepcopy(theta)
                best_model = cp.deepcopy(network)
                print(f"\nBest validation error: {val_loss}\nBest hyperparameters: {theta}\nBest accuracy: {accuracy}\n")
            
            if task == "cup":
                model = {
                    'theta': best_theta,
                    'model': best_model,
                    'validation_error': val_losses[-1],
                    # AGGIUNGERE GRAFICI
                }
                if len(best_hyperparams) == 5:
                    best_hyperparams.pop(0)
                    best_hyperparams.append(model)
        
            left_combinations -= 1
            print(f"Combinations left: {left_combinations}\n")
        
        return best_theta, best_model, best_hyperparams ,val_losses[-1]
 
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

        print(dictionary_combination)
    
   

    return result_combinations

def model_assessment(final_model, test_X, test_y):
    # Compute accuracy on test set
    test_predictions = final_model.predict(test_X)
    accuracy = final_model.compute_accuracy(test_y, test_predictions)
    print(" Accuracy TEST: ", accuracy)