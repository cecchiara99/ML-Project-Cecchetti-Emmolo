import pandas as pd
import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = sigmoid(np.dot(inputs, self.weights))
        return self.output
    
    def backward(self, error, learning_rate):
        delta = error * sigmoid_derivative(self.output)
        self.weights += self.inputs.T.dot(delta) * learning_rate
        return delta

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(input_size, hidden_size)
        self.output_layer = Layer(hidden_size, output_size)
    
    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        return self.output_layer.forward(hidden_output)
    
    def backward(self, error, learning_rate):
        output_error = self.output_layer.backward(error, learning_rate)
        self.hidden_layer.backward(output_error.dot(self.output_layer.weights.T), learning_rate)
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            predicted_output = self.forward(X)
            error = y - predicted_output
            self.backward(error, learning_rate)
            
            # Stampa l'errore ogni 1000 epoche
            if epoch % 1000 == 0:
                print(f'Epoch: {epoch}, Error: {np.mean(np.abs(error))}')

# Funzione di attivazione sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivata della funzione di attivazione sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Funzione per l'One-Hot-Encoding
def one_hot_encoding(data):
    encoded_data = pd.get_dummies(data, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
    return encoded_data

# Carica i dati dal file CSV
data = pd.read_csv(percorso_file_train_1, delimiter=' ', header=None, names=['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'ID'])

# Applica One-Hot-Encoding ai dati
encoded_data = one_hot_encoding(data[['a1', 'a2', 'a3', 'a4', 'a5', 'a6']])
X = np.hstack([encoded_data.values, np.ones((encoded_data.shape[0], 1))])  # Aggiungi una colonna di bias
y = data['target'].values.reshape(-1, 1)

# Inizializza la rete neurale
input_size = X.shape[1]
hidden_size = 8
output_size = 1
learning_rate = 0.01
epochs = 1000

neural_network = NeuralNetwork(input_size, hidden_size, output_size)

# Addestra la rete neurale con Stochastic Gradient Descent (SGD)
neural_network.train(X, y, epochs, learning_rate)