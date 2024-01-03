import numpy as np
import pandas as pd

class Layer:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid',
                 learning_rate=0.01, momentum=0.9, weight_decay=0.001):
        # Inizializzazione dei pesi e dei bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Inizializzazione dei pesi e dei bias
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        # Inizializzazione dei termini di momentum
        self.momentum_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.momentum_bias_hidden = np.zeros_like(self.bias_hidden)
        self.momentum_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.momentum_bias_output = np.zeros_like(self.bias_output)

        # Funzione di attivazione
        self.activation = self.sigmoid if activation == 'sigmoid' else self.relu
        self.output_activation = self.softmax if output_size > 1 else self.sigmoid

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    

