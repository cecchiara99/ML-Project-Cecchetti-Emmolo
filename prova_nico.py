import pandas as pd
import numpy as np
from neural_network import NeuralNetwork
from read_data import *
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

data_X, data_y = read_monk(percorso_file_train_1)

print("Monk-shape: ", data_X.shape)
print("Targets-shape: ", data_y.shape)

input_size = data_X.shape[1]
output_size = data_y.shape[1]
activation_hidden = "sigmoid"
activation_output = "tanh"


# Train the model on the training set and select the best model
best_model = model_selection(input_size, output_size, activation_hidden, activation_output, data_X, data_y, K=3)

# Assess the performance of the best model on the test set
#test_error = model_assessment(best_model, test_data)
#print(f"Final Test Error: {test_error}")

# PER MEE
"""
def evaluate(self, inputs, targets):
    _, predictions = self.forward_propagation(inputs)
    mee = np.mean(np.sqrt((targets - predictions)**2))
    # oppure np.sqrt(np.mean((targets - predictions)**2))
    # oppure np.mean(np.sqrt(np.sum((targets - predictions)**2, axis=1)))
    return mee
"""


hyperparameters_ranges =  {
    'hidden_size': (2, 5, 1),      # Specify range (lower_limit, upper_limit, step)
    'learning_rate': (0.1, 0.9, 0.01),  # Specify range (lower_limit, upper_limit, step)
    'epochs': (200, 1000, 200),        # Specify range (lower_limit, upper_limit, step)
    'batch_size': (128, 128, 64),      # Specify range (lower_limit, upper_limit, step)
    'momentum': (0.5, 0.9, 0.01),        # Specify range (lower_limit, upper_limit, step)
    'lambda_reg': (0.001, 0.1, 0.001),  # Specify range (lower_limit, upper_limit, step)
}