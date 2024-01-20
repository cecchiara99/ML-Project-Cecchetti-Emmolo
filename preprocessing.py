# Code Standard 8: Comments and Documentation

import pandas as pd
import numpy as np

# Specify the paths to your training and test files
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

percorso_file_train_cup = './cup+problem/ML-CUP23-TR.csv'

class Preprocessing:
    def preprocessing(path):
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

    
    def preprocessing_cup(path):
        # Read the training dataset
        col_names = ['Id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y', 'target_z']
        cup_dataset = pd.read_csv(path, sep=',', names=col_names, skiprows=range(7), usecols=range(0, 11))
        cup_dataset.set_index('Id', inplace=True) 
        targets = pd.read_csv(path, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 14))

        print(cup_dataset)
        print(targets)

        return cup_dataset, targets

# Preprocessing of the training dataset
"""monk_dataset_array, targets_array = Preprocessing.preprocessing(percorso_file_train_1)

#print(monk_dataset_array)
#print(targets_array)
print("Monk-shape: ", monk_dataset_array.shape)
print("Targets-shape: ", targets_array.shape)

# Preprocessing of the training dataset
cup_dataset, targets = Preprocessing.preprocessing_cup(percorso_file_train_cup)

print("Cup-shape: ", cup_dataset.shape)
print("Targets-shape: ", targets.shape)"""




