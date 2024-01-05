import pandas as pd
import numpy as np
from Layer import Layer
from preprocessing import Preprocessing

class NeuralNetwork:
    def __init__(self):
        """
        Initialize the neural network.

        Args:
            None
        """
        self.layers = []

    def add_layer(self, input_size, output_size):
        """
        Add a layer to the neural network.

        Args:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.

        Returns:
            None
        """

        # Create a new layer and append it to the list of layers
        layer = Layer(input_size, output_size)
        self.layers.append(layer) 

    def forward(self, inputs):
        """
        Compute the forward pass of the neural network.

        Args:
            inputs (ndarray): The input.

        Returns:
            ndarray: The output of the neural network.
        """

        # Compute the forward pass through each layer
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs
        
    def backward(self, target, learning_rate):
        """
        Backpropagate the error through the network.

        Args:
            target (ndarray): The target.
            learning_rate (float): The learning rate.

        Returns:
            None
        """

        # Compute the gradient of the error with respect to each weight
        output_layer = self.layers[-1] # Get the output layer
        output_error = target - output_layer.outputs # Compute the error
        output_gradient = output_error * output_layer.activation_derivative(output_layer.outputs) # Compute the gradient of the error with respect to the output

        # Backpropagate the gradient through the network
        for layer in reversed(self.layers):
            output_gradient = layer.backward_pass(output_gradient, learning_rate)
    
    def train(self, input_data, targets, epochs, learning_rate):
        """
        Train the neural network with Stochastic Gradient Descent (SGD).

        Args:
            input_data (ndarray): The input data.
            targets (ndarray): The targets.
            epochs (int): The number of epochs.
            learning_rate (float): The learning rate.

        Returns:
            None
        """

        # Train the neural network for the specified number of epochs
        for epoch in range(epochs):
            predicted_output = self.forward(input_data) # Forward pass
            error = targets - predicted_output # Compute the error
            self.backward(error, learning_rate) # Backward pass
            
            # Stampa l'errore ogni 1000 epoche
            if epoch % 1000 == 0:
                print(f'Epoch: {epoch}, Error: {np.mean(np.abs(error))}') # Print the error

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
