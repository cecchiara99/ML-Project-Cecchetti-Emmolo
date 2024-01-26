import pandas as pd
import numpy as np

def read_monk(path):
    """
    Preprocesses the data from the specified file path.

    Args:
        path (str): The path to the data file.

    Returns:
        tuple: A tuple containing the preprocessed data array and labels array.
    """
    # Read the training dataset
    col_names = ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk_dataset = pd.read_csv(path, sep=' ', header=None, names=col_names)
    monk_dataset.set_index('Id', inplace=True) 
    targets = monk_dataset.pop('target')

    # One-Hot-Encoding for all columns except the target column
    monk_dataset_encoded = pd.get_dummies(monk_dataset, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'], dtype=float)

    # Convert the DataFrame to a NumPy array
    monk_dataset_array = monk_dataset_encoded.to_numpy(dtype=np.float32)

    # Convert the labels to a NumPy array 
    targets_array = targets.to_numpy(dtype=np.float32).reshape(-1, 1) # reshape for having (n,1) instead of (n,)

    return monk_dataset_array, targets_array


def read_cup(path_tr, path_ts):
    # Read the training dataset
    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y', 'target_z']
    train_dataset = pd.read_csv(path_tr, sep=',', names=col_names, skiprows=range(7), usecols=range(0, 11))
    train_dataset.set_index('Id', inplace=True) 
    targets = pd.read_csv(path_tr, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 14))

    train_dataset_array = train_dataset.to_numpy(dtype=np.float32)
    targets_array = targets.to_numpy(dtype=np.float32)

    # Take the first 80% of the training dataset as training set and the remaining 20% as internal test set
    n = int(0.8 * len(train_dataset_array))
    train_dataset_array = train_dataset_array[:n]
    targets_array = targets_array[:n]
    internal_test_dataset_array = train_dataset_array[n:]
    internal_test_targets_array = targets_array[n:]

    # Read the test dataset (blind)
    col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']
    blind_test_dataset = pd.read_csv(path_ts, sep=',', names=col_names, skiprows=range(7), usecols=range(0, 11))
    blind_test_dataset.set_index('Id', inplace=True)

    blind_test_dataset_array = blind_test_dataset.to_numpy(dtype=np.float32)

    return train_dataset_array, targets_array, internal_test_dataset_array, internal_test_targets_array, blind_test_dataset_array




