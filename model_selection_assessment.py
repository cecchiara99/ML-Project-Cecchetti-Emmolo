import numpy as np
import copy as cp
from neural_network import NeuralNetwork
from itertools import product
from sklearn.model_selection import train_test_split
from utils import *
import json
import shutil

def model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, task, test_X, test_y, type_selection = "k-fold"):

    len_data = data_X.shape[0]
    test_losses = []
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

    hyperparameters = [{'hidden_size': 3, 'learning_rate': 0.9, 'epochs': 400, 'batch_size': 64, 'momentum': 0.9, 'lambda_reg': 0.001, 'w_init_limit': [-0.3, 0.3]}]

    if type_selection == "k-fold":
        K = 5
        print(f"\nINIZIO K-fold cross validation, K = {K} \n")

        if task != "cup":
            best_theta, best_model,best_validation_error, best_accuracy, losses, test_losses, accuracies, test_accuracies = k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, K, task, test_X, test_y, patience=10)
        else:
            best_theta, best_model,best_validation_error, best_mee, mees_validation, mees_test, mees = k_fold_cross_validation(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, K, task, test_X, test_y, patience=10)

    elif type_selection == "hold-out":
        print("\nINIZIO Hold-out\n")
        if task != "cup":
            best_theta, best_model,best_validation_error, losses, test_losses, accuracies, test_accuracies = hold_out(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, task, test_X, test_y, patience=10)
        else:
            best_theta, best_model,best_validation_error, mees_validation, mees_test, mees = hold_out(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameters, task, test_X, test_y, patience=10)
        
    
    print(f"Best hyperparameters: {best_theta}")

    if task != "monk3" and task != "cup":
        print("END MODEL SELECTION, START RETRAINING...\n")
        # Train the model on the whole training set using the best hyperparameters
        losses, test_losses, accuracies, test_accuracies = best_model.train_monk(data_X, data_y, test_X, test_y, task)
    
    if task == 'cup':
        mees, mees_test = best_model.train_cup(data_X, data_y, test_X, test_y, task)
        
    
    if task != "cup":
        # Compute accuracy on validation set
        val_predictions = best_model.predict(data_X)
        accuracy = best_model.compute_accuracy(data_y, val_predictions)
        print("Final accuracy (after retraining): ", accuracy)
    else:
        mee = mean_euclidean_error(data_y, best_model.predict(data_X))
        print("\nFinal MEE (after retraining): ", mee)
    
    # Copy the best model
    final_model = cp.deepcopy(best_model)
    
    
    
    #PLOT DEI VARI GRAFICI A SECONDA DEL TASK
    if task == 'cup':

        plt.plot(range(0, final_model.epochs), mees, label='Validation', color='blue')
        plt.plot(range(0, final_model.epochs), mees_test, label='Test', color='red')

        plt.title('MEE curve')
        plt.xlabel('Epochs')
        plt.ylabel('MEE')
        plt.legend()
        plt.savefig('./mee_curve.png')
        plt.close()
    
    else:
        plt.plot(range(0, final_model.epochs), accuracies, label='Accuracy_Training', color='blue')
        plt.plot(range(0, final_model.epochs), test_accuracies, label='Test Accuracy', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig('accuracy_curve.png')  
        plt.close()

        plt.plot(range(0, final_model.epochs), losses, label='Accuracy_Training', color='blue')
        plt.plot(range(0, final_model.epochs), test_losses, label='Test Accuracy', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Learning Curve')
        plt.legend()
        plt.savefig('learning_curve.png')  
        plt.close()

    if task == "cup":
        source_file_mee = './mee_curve.png'
        destination_file_mee = './model_graphs/' + task + '_'+ type_selection  + '_mee.jpg'
        with open(source_file_mee, 'rb') as f:
            with open(destination_file_mee, 'wb+') as f1:
                shutil.copyfileobj(f, f1)
    else:
        source_file_accuracy = './accuracy_curve.png'
        destination_file_accuracy = './model_graphs/' + task + '_'+ type_selection  + '_accuracy.jpg'
        with open(source_file_accuracy, 'rb') as f:
            with open(destination_file_accuracy, 'wb+') as f1:
                shutil.copyfileobj(f, f1)
        # Specify the source and destination file paths
        source_file_learning = './learning_curve.png'
        destination_file_learning = './model_graphs/' + task+ '_'+ type_selection + '_learning_curve.jpg'

        # Copy and rename the image file
        with open(source_file_learning, 'rb') as f:
            with open(destination_file_learning, 'wb+') as f1:
                shutil.copyfileobj(f, f1)

    if task == "monk1" or task == "monk2" or task == "monk3":
        # Create a dictionary containing the model info
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
    else:
        # Create a dictionary containing the model info
        model_info_cup = {
            'theta': best_theta,
            'model_selection': type_selection,
            'validation_error': best_validation_error,
            'mee': mee,
            'img_mee_curve': destination_file_mee,
        }

        # Save the model info in a json file
        with open('./models_info/'+ task+ '_' + type_selection +'_model_info.json', 'w+') as outfile:
            json.dump(model_info_cup, outfile)

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

    best_hyperparams = []

    left_combinations = len(hyperparams)

    # Cycle for grid search
    for theta in hyperparams:
        tot_validation_error = 0.0
        
        accuracies_validation = []
        mees_validation = []
  
        indices = np.random.permutation(len(data_y))
        X_shuffled = data_X[indices]
        y_shuffled = data_y[indices]

        # Cycle for K-fold cross validation
        for k in range(K):
            # Split the data into training and validation sets
            training_X, training_y, validation_X, validation_y = split_data_into_folds(X_shuffled, y_shuffled, K, k)
            
            # Train the model on the training set
            network = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **theta)
            
            if task != "cup":
                losses, test_losses, accuracies, test_accuracies = network.train_monk(training_X, training_y, test_X, test_y, task)
            else:
                mees, mees_test = network.train_cup(training_X, training_y, test_X, test_y, task)
            
            # Evaluate the model on the validation set
            validation_error = network.evaluate(validation_X, validation_y, task)
            
            val_predictions = network.predict(validation_X)
            
            if task != "cup":
                accuracies_validation.append(network.compute_accuracy(validation_y, val_predictions))
            else:
                mees_validation.append(mean_euclidean_error(validation_y, val_predictions))
            
            tot_validation_error += validation_error


        # Compute the average validation error
        avg_validation_error = tot_validation_error / K

        print(f"\nValidation error: {avg_validation_error}\n")
        print(f"\nBest validation error: {best_validation_error}\n")
        

        # Update best hyperparameter and best model if the current ones are better
        if avg_validation_error < best_validation_error:
            print("BEST VALIDATION ERROR UPDATED\n")
            best_validation_error = avg_validation_error
            print(f"\nNEW BEST VALIDATION ERROR: {best_validation_error}\n")

            best_theta = cp.deepcopy(theta)
            best_model = cp.deepcopy(network)

            if task != "cup":
                best_accuracy = np.mean(accuracies_validation[:len(validation_y)]) 
                print(f"\nNEW BETTER!!\nBest validation error: {best_validation_error}\nBest hyperparameters: {theta}\nBest accuracy: {best_accuracy}\n")

            else:
                best_mee = best_validation_error
                print(f"\nNEW BETTER!!\nBest validation error: {best_validation_error}\nBest hyperparameters: {theta}\nBest MEE: {best_mee}\n")
            
            
            # Save the 5 best model info in a json file
            if task == "cup":
                model = {
                    'theta': best_theta,
                    'validation_error (mee)': best_validation_error,
                    'test_error (mee)': mees_test[-1],
                    'K-fold': K,
                    'mees': mees,
                    'mees_test': mees_test,
                    
                }
                if len(best_hyperparams) >= 5:
                    best_hyperparams.pop(0)
                    best_hyperparams.append(model)
                else:
                    best_hyperparams.append(model)
            
            left_combinations -= 1
            print(f"\nCombinations left: {left_combinations}\n")
    
    if task == "cup":
        # Save the model info in a json file
        with open('./models_best_cup/'+ task+ '_k-fold' +'_model_info.json', 'w+') as outfile:
            json.dump(best_hyperparams, outfile)
    
    if task == "cup":
        return best_theta, best_model,best_validation_error, best_mee, mees_validation, mees_test, mees
    else:
        return best_theta, best_model,best_validation_error, best_accuracy, losses, test_losses, accuracies, test_accuracies
    
def hold_out(input_size, output_size, activation_hidden, activation_output, data_X, data_y, hyperparameter, task, test_X, test_y, patience):

        network = None
        best_theta = None
        best_model = None
        left_combinations = len(hyperparameter)
        
        best_validation_error = float('inf')
        
        best_hyperparams = []
        
        for theta in hyperparameter:
            val_losses, accuracies_validation, mees_validation, mees_test = [], [], [], []
            
            X_train, y_train, X_val, y_val = split_data(data_X, data_y, 0.8)

            indices = np.random.permutation(len(y_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            network = NeuralNetwork(input_size, output_size, activation_hidden, activation_output, **theta)
            
            if task != "cup":
                losses, test_losses, accuracies, test_accuracies = network.train_monk(X_train, y_train, test_X, test_y, task)
            else:
                mees, mees_test = network.train_cup(X_train, y_train, test_X, test_y, task)
                
            # Evaluate on validation set
            val_loss = network.evaluate(X_val, y_val, task)
            val_losses.append(val_loss)

            # Compute accuracy on validation set
            val_predictions = network.predict(X_val)

            if task != "cup":
                accuracies_validation.append(network.compute_accuracy(y_val, val_predictions))

            else:
                mees_validation = cp.deepcopy(val_losses)

            # Update best hyperparameter and best model if the current ones are better
            if len(val_losses) > 1:
                if val_loss < val_losses[-2]:
                    best_theta = cp.deepcopy(theta)
                    best_model = cp.deepcopy(network)

                    best_validation_error = val_loss
                    
                    if task != "cup":
                        print(f"\nNEW BETTER!!\nBest validation error: {val_loss}\nBest hyperparameters: {theta}\nBest accuracy: {accuracies_validation[-1]}\n")
                    else:
                        print(f"\nNEW BETTER!!\nBest validation error: {val_loss}\nBest hyperparameters: {theta}\nBest MEE: {mees_validation[-1]}\n")
                        
                        # Save the 5 best model info in a json file
                        
                        model = {
                            'theta': best_theta,
                            'validation_error (mee)': best_validation_error,
                            #aggiungi grafici
                        }
                        if len(best_hyperparams) >= 5:
                            best_hyperparams.pop(0)
                            best_hyperparams.append(model)
                        else:
                            best_hyperparams.append(model)

            else:
                best_theta = cp.deepcopy(theta)
                best_model = cp.deepcopy(network)
                best_validation_error = val_loss
                if task == "cup":
                    model = {
                                'theta': best_theta,
                                'validation_error (mee)': best_validation_error,
                                
                                #aggiungi grafici
                    }

                    best_hyperparams.append(model)

                
            
        
            left_combinations -= 1
            print(f"Combinations left: {left_combinations}\n")
        
        if task == "cup":
            # Save the model info in a json file
            with open('./models_best_cup/'+ task+ '_hold-out' +'_model_info.json', 'w+') as outfile:
                json.dump(best_hyperparams, outfile)
    
        if task == "cup":
            return best_theta, best_model,best_validation_error, mees_validation, mees_test, mees
        else:
            return best_theta, best_model,best_validation_error, losses, test_losses, accuracies, test_accuracies
 
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
    
    print("\nHyperparameters combinations generated\n")
    print(f"Number of combinations: {len(result_combinations)}\n")

    return result_combinations

def model_assessment(final_model, test_X, test_y, task):
    test_predictions = final_model.predict(test_X)
    
    if task != "cup":
        # Compute accuracy on test set 
        accuracy = final_model.compute_accuracy(test_y, test_predictions)
        print("\nAccuracy TEST: ", accuracy)
    else:
        mee = mean_euclidean_error(test_y, test_predictions)
        print("\nMEE TEST: ", mee)