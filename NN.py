import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import anderson
import numpy as np
from statsmodels.stats.diagnostic import lilliefors

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation_function):
        layer = Layer(input_size, output_size, activation_function)
        self.layers.append(layer)

    def forward_pass(self, input_data):
        current_layer_output = input_data

        for layer in self.layers:
            # Multiply by weights and add bias
            weighted_sum = np.dot(current_layer_output, layer.weights) + layer.bias
            # Apply activation function
            current_layer_output = layer.activation_function(weighted_sum)

        final_output = current_layer_output
        return final_output

def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))



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



# Example usage:
# Create a neural network
my_nn = NeuralNetwork()

# Add layers to the neural network
my_nn.add_layer(input_size=4, output_size=3, activation_function=sigmoid)
my_nn.add_layer(input_size=3, output_size=2, activation_function=sigmoid)

# Perform a forward pass with some input data
input_data = np.array([[0.1, 0.2, 0.3, 0.4]])
output = my_nn.forward_pass(monk_dataset_array)

print("Final output:", output)

