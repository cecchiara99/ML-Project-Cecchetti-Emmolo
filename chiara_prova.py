import numpy as np
from preprocessing import Preprocessing
import matplotlib.pyplot as plt

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, epochs=1000, batch_size=32, momentum=0.9, lambda_reg=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.lambda_reg = lambda_reg

        self.weights_input_hidden, self.biases_hidden, self.weights_hidden_output, self.biases_output = self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(42)
        weights_input_hidden = np.random.uniform(low=0.1, high=0.9, size=(self.input_size,self.hidden_size))
        biases_hidden = np.zeros((1, self.hidden_size))
        weights_hidden_output = np.random.uniform(low=0.1, high=0.9, size=(self.hidden_size, 1))
        biases_output = np.zeros((1, 1))
        return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

    def tanh(self, x):
        return np.tanh(x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.biases_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.biases_output
        output = self.tanh(output_layer_input)

        return hidden_layer_output, output

    def calculate_loss(self, y, y_pred):
        m = len(y)
        loss = np.mean((y_pred - y) ** 2)
        regularization_term = (self.lambda_reg / (2 * m)) * (
            np.sum(self.weights_input_hidden ** 2) + np.sum(self.weights_hidden_output ** 2)
        )
        return loss + regularization_term

    def backward_propagation(self, X, y, hidden_layer_output, output):
        m = len(y)
        error_output = output - y

        output_delta = error_output * self.tanh_derivative(output)
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(hidden_layer_output)

        weights_hidden_output_gradient = hidden_layer_output.T.dot(output_delta) / m
        biases_output_gradient = np.sum(output_delta, axis=0, keepdims=True) / m

        weights_input_hidden_gradient = X.T.dot(hidden_layer_delta) / m
        biases_hidden_gradient = np.sum(hidden_layer_delta, axis=0, keepdims=True) / m

        weights_hidden_output_gradient += (self.lambda_reg / m) * self.weights_hidden_output
        weights_input_hidden_gradient += (self.lambda_reg / m) * self.weights_input_hidden

        # Update with momentum
        self.weights_hidden_output -= self.momentum * self.learning_rate * weights_hidden_output_gradient
        self.biases_output -= self.momentum * self.learning_rate * biases_output_gradient
        self.weights_input_hidden -= self.momentum * self.learning_rate * weights_input_hidden_gradient
        self.biases_hidden -= self.momentum * self.learning_rate * biases_hidden_gradient


    def train(self, X, y):
        m = len(y)
        losses = []

        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                hidden_layer_output, output = self.forward_propagation(X_batch)
                loss = self.calculate_loss(y_batch, output)
                self.backward_propagation(X_batch, y_batch, hidden_layer_output, output)
            losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        
        plt.plot(range(0, self.epochs), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.savefig('learning_curve.png')  
        plt.close()

"""    def predict(self, X):
        _, output = self.forward_propagation(X)
        return output"""

percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

X_train, y_train = Preprocessing.preprocessing(percorso_file_train_1)

print("Monk-shape: ", X_train.shape)
print("Targets-shape: ", y_train.shape)

input_size = X_train.shape[1]
hidden_size = 4

nn = NeuralNetwork(input_size, hidden_size, learning_rate=0.1, epochs=1000, batch_size=32, momentum=0.5, lambda_reg=0.01)
nn.train(X_train, y_train)


