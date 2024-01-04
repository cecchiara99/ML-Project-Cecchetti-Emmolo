import pandas as pd
import numpy as np
from Layer import Layer
from preprocessing import Preprocessing

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, hidden_size, output_size):
        layer = Layer(input_size, hidden_size, output_size)
        self.layers.append(layer)

    def forward_pass(self, input_data):
        current_layer_output = input_data

        for layer in self.layers:
            # Multiply by weights and add bias
            weighted_sum = np.dot(current_layer_output, layer.weights_input_hidden) + layer.bias_hidden
            # Apply activation function
            current_layer_output = layer.activation(weighted_sum)

        final_output = current_layer_output
        return final_output

percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

# Preprocessing of the training dataset
monk_dataset_array, labels_array = Preprocessing.preprocessing(percorso_file_train_1)

# Display the shapes of the preprocessed arrays
print("Monk1-shape: ", monk_dataset_array.shape)
print("Labels-shape: ", labels_array.shape)

