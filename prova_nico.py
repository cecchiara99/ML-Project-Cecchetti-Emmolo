import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import anderson
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from NN import NeuralNetwork
from Layer import Layer
from preprocessing import Preprocessing


# Specifica i percorsi dei tuoi file di addestramento e di test
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'



# Example usage:
# Create a neural network
my_nn = NeuralNetwork()

# Add layers to the neural network
my_nn.add_layer(input_size=17, output_size=4)
my_nn.add_layer(input_size=4, output_size=2)

# Perform a forward pass with some input data
input_data,_ = Preprocessing.preprocessing(percorso_file_train_1)
output = my_nn.forward(input_data)

print("Final output:", output)