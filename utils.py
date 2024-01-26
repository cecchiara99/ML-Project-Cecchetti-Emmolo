
import numpy as np
import pandas as pd
from datetime import date

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
    #mee = np.mean(np.sqrt((targets - predictions)**2))
    #mee = np.sqrt(np.mean((targets - predictions)**2))
    #mee = np.mean(np.sqrt(np.sum((targets - predictions)**2, axis=1)))
    #mee = np.linalg.norm(np.subtract(predicted, target))
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
    team_name = "team_name"
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