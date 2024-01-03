import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import anderson
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from Layer import Layer

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



