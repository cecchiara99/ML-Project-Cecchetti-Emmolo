import pandas as pd
import numpy as np
from chiara_prova import NeuralNetwork
from Layer import Layer
from preprocessing import Preprocessing
from model_selection import *

"""
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
nn.add_layer(input_size, hidden_size, "sigmoid") # hidden layer
nn.add_layer(hidden_size, output_size, "tanh") # output layer

# Train the neural network with Stochastic Gradient Descent (SGD)
nn.train(input_data, targets, epochs=1000, learning_rate=0.01, momentum=0.9, weight_decay=0.01)
"""


percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

data_X, data_y = Preprocessing.preprocessing(percorso_file_train_1)

print("Monk-shape: ", data_X.shape)
print("Targets-shape: ", data_y.shape)

input_size = data_X.shape[1]
output_size = data_y.shape[1]
activation_hidden = "sigmoid"
activation_output = "sigmoid"


# Train the model on the training set and select the best model
best_model = model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, K=3)

# Assess the performance of the best model on the test set
#test_error = model_assessment(best_model, test_data)
#print(f"Final Test Error: {test_error}")