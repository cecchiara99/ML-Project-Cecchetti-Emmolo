import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import anderson
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from NN import NeuralNetwork



# Specifica i percorsi dei tuoi file di addestramento e di test
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

"""
# Carica i file di addestramento e di test in DataFrame separati
df_train_1 = pd.read_csv(percorso_file_train_1)
df_train_2 = pd.read_csv(percorso_file_train_2)
df_train_3 = pd.read_csv(percorso_file_train_3)
"""

"""
# Leggi il dataset di addestramento 1
col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
monk_dataset = pd.read_csv(percorso_file_train_1, sep=' ', names=col_names)
monk_dataset.set_index('Id', inplace=True)
labels = monk_dataset.pop('class')

# Seleziona solo le colonne numeriche per la normalizzazione
numeric_columns = monk_dataset.columns
numeric_data = monk_dataset[numeric_columns]

# Normalizza manualmente le colonne numeriche
normalized_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())

# Riunisci il dataset normalizzato con le etichette
monk_dataset_normalized = pd.concat([normalized_data, labels], axis=1)

# Stampa il dataset normalizzato
print("Dataset normalizzato:")
print(monk_dataset_normalized)
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(input_data, weights, biases):
    # Inizializzazione con i dati di input
    current_layer_output = input_data

    # Iterazione attraverso gli strati
    for layer in NeuralNetwork.layers:
        # Moltiplicazione dei pesi e somma ponderata
        weighted_sum = current_layer_output.dot(layer.weights) + layer.bias
        
        # Funzione di attivazione
        current_layer_output = sigmoid(weighted_sum)
    
    # Output finale
    final_output = current_layer_output
    return final_output


# Leggi il dataset di addestramento 1
col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
monk_dataset = pd.read_csv(percorso_file_train_1, sep=' ', names=col_names)
monk_dataset.set_index('Id', inplace=True)
labels = monk_dataset.pop('class')

# Effettuare il One-Hot-Encoding per tutte le colonne
monk_dataset_encoded = pd.get_dummies(monk_dataset, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'], dtype=float)

# Riunisci il dataset codificato con le etichette
monk_dataset_encoded['class'] = labels

# Stampare il DataFrame risultante
# print(monk_dataset_encoded)

# Converte il DataFrame in un array NumPy
monk_dataset_array = monk_dataset_encoded.to_numpy(dtype=np.float32)
print(monk_dataset_array[0])

output = forward_pass(monk_dataset_array[0], NeuralNetwork., NeuralNetwork.biases)
print(output)

