import pandas as pd
import numpy as np
from Layer import NeuralNetwork

# Specifica i percorsi dei tuoi file di addestramento e di test
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

# Read the training dataset 1
col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
monk_dataset = pd.read_csv(percorso_file_train_1, sep=' ', names=col_names)
monk_dataset.set_index('Id', inplace=True)
labels = monk_dataset.pop('class')

# One-Hot-Encoding for all columns
monk_dataset_encoded = pd.get_dummies(monk_dataset, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'], dtype=float)

# Reunite the encoded dataset with the labels
monk_dataset_encoded['class'] = labels

# Print the resulting DataFrame
print(monk_dataset_encoded)

# Convert the DataFrame to a NumPy array
monk_dataset_array = monk_dataset_encoded.to_numpy(dtype=np.float32)
print(monk_dataset_array[0])


# dividere fra features e target value dopo encoding