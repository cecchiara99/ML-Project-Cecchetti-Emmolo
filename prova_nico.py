import pandas as pd
import numpy as np
from NN import NeuralNetwork
from Layer import Layer
from preprocessing import Preprocessing


# Specified the paths to your training and test files
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

# Preprocessing
input_data, targets = Preprocessing.preprocessing(percorso_file_train_1)

# Initialize the neural network
input_size = input_data.shape[1]
hidden_size = 8
output_size = 1


# Create a neural network
nn = NeuralNetwork()

# Add layers to the neural network
nn.add_layer(input_size, hidden_size)
nn.add_layer(hidden_size, output_size)

# Train the neural network with Stochastic Gradient Descent (SGD)
nn.train(input_data, targets, epochs=1000, learning_rate=0.01, momentum=0.9, weight_decay=0.001)