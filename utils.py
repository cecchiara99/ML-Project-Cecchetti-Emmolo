
import numpy as np
import pandas as pd


def mean_squared_error(y_true, y_pred):
    """
    Calcola il Mean Squared Error tra i valori veri e le previsioni.

    Args:
        y_true (numpy.ndarray): Array con i valori veri.
        y_pred (numpy.ndarray): Array con le previsioni.

    Returns:
        float: Mean Squared Error (MSE).
    """
    error = y_true - y_pred
    mse = np.mean(np.square(error))
    return mse

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