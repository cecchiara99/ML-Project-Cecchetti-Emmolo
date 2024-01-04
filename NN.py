import pandas as pd
import numpy as np
from Layer import Layer
from preprocessing import Preprocessing

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size):
        layer = Layer(input_size, output_size)
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs
        

"""
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

# Preprocessing of the training dataset
monk_dataset_array, targets_array = Preprocessing.preprocessing(percorso_file_train_1)

# Display the shapes of the preprocessed arrays
print("Monk1-shape: ", monk_dataset_array.shape)
print("Labels-shape: ", targets_array.shape)
"""
