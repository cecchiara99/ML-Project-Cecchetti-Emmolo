
import numpy as np
import pandas as pd
from datetime import date
from matplotlib import pyplot as plt

def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error between the true and predicted values.

    :param y_true: the true values
    :param y_pred: the predicted values

    :return: the mean squared error
    """
    
    error = y_true - y_pred
    mse = np.mean(np.square(error))
    return mse


def mean_euclidean_error(y_true, y_pred):
    """
    Compute the mean euclidean error between the true and predicted values.

    :param y_true: the true values
    :param y_pred: the predicted values

    :return: the mean euclidean error
    """
    
    error = y_true - y_pred
    mee = np.mean(np.linalg.norm(error, axis=1))
    return mee


def normalize_data(data_matrix, labels):
    """
    Normalize the dataset.

    Args:
        data_matrix (pd.DataFrame): Matrix containing the input features.
        labels (pd.Series): Series containing the class labels.

    Returns:
        pd.DataFrame: Normalized dataset.
    """
    
    # Seleziona solo le colonne numeriche per la normalizzazione
    numeric_columns = data_matrix.columns
    numeric_data = data_matrix[numeric_columns]

    # Normalizza manualmente le colonne numeriche
    normalized_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())

    # Riunisci il dataset normalizzato con le etichette
    normalized_dataset = pd.concat([normalized_data, labels], axis=1)

    return normalized_dataset


def split_data(data_X, data_y, percentage):

    """
    Split the data into different sets, basing on the percentage.

    :param data: the dataset
    :param percentage: the percentage of the split

    :return: the splitted data
    """

    n = int(percentage * len(data_y))
    
    # Split the data
    first_X = data_X[:n]
    first_y = data_y[:n]
    second_X = data_X[n:]
    second_y = data_y[n:]

    return first_X, first_y, second_X, second_y


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


def initialize_best_results(n):
    best_results = []

    for i in range(n):
        best_results.append({
            'theta': None,
            'model': None,
            'validation_error': float('inf')
        })
    
    return best_results


def create_cup_csv(outputs):
    """
    Create the CSV file for the ML-CUP23 competition.

    :param outputs: the predictions of the model

    :return: None
    """

    # Create a DataFrame with the predictions and the Ids
    df = pd.DataFrame(outputs, columns=['output_x', 'output_y', 'output_z'])
    df.insert(0, 'Id', range(1, len(outputs)+1))
    
    # Create the CSV file
    team_name = "gradient_decent"
    submission_date = date.today().strftime("%d/%m/%Y")
    output_file_path = f"{team_name}_ML-CUP23-TS.csv"
    with open(output_file_path, 'w', newline='') as f:
        # Initial rows
        f.write(f"# Chiara Cecchetti, Nicola Emmolo\n")
        f.write(f"# {team_name}\n")
        f.write(f"# ML-CUP23\n")
        f.write(f"# Submission Date ({submission_date})\n")

        # Write the DataFrame to the CSV file
        df.to_csv(f, index=False, header=False, sep=',')

    print(f"File CSV '{output_file_path}' created")


def plot_graphs_monk(losses, test_losses, accuracies, test_accuracies, epochs):
    plt.plot(range(0, epochs), losses, label='Training', color='blue')
    plt.plot(range(0, epochs), test_losses, label='Test', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('learning_curve.png')  
    plt.close()

    plt.plot(range(0, epochs), accuracies, label='Accuracy Training', color='blue')
    plt.plot(range(0, epochs), test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig('accuracy_curve.png')  
    plt.close()


def plot_graphs_cup(mees, mees_test, epochs):
    plt.plot(range(0, epochs), mees, label='Validation', color='blue')
    plt.plot(range(0, epochs), mees_test, label='Test', color='red')

    plt.title('MEE curve')
    plt.xlabel('Epochs')
    plt.ylabel('MEE')
    plt.legend()
    plt.savefig('./mee_curve.png')
    plt.close()


# Set hyperparameters ranges for MONK or CUP
def set_hyperparameters_ranges(task, len_data):
    if task == "monk":
        # Set hyperparameters ranges for MONK
        hyperparameters_ranges =  {
            # Specify range (lower_limit, upper_limit, step)
            'hidden_size': (3, 4, 1),           
            'learning_rate': (0.1, 0.9, 0.1),
            'epochs': (300, 500, 100),
            'batch_size': [64, len_data],
            'momentum': (0.5, 0.9, 0.1),
            'lambda_reg': [0.001, 0.01, 0.1],
            'w_init_limit': [[-0.3, 0.3],[-0.2, 0.2],[-0.1,0.1]]
        }
    elif task == "cup":
        # Set hyperparameters ranges for CUP
        hyperparameters_ranges = {
            # Specify range (lower_limit, upper_limit, step)
            'hidden_size': (3, 4, 1),           
            'learning_rate': (0.1, 0.9, 0.1),
            'epochs': (300, 500, 100),
            'batch_size': [64, len_data],
            'momentum': (0.5, 0.9, 0.1),
            'lambda_reg': [0.001, 0.01, 0.1],
            'w_init_limit': [[-0.3, 0.3],[-0.2, 0.2],[-0.1,0.1]]
        }
    else:
        print("Error: task not recognized")
        return None

    return hyperparameters_ranges