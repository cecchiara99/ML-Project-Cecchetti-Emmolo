import numpy as np
from preprocessing import Preprocessing
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_hidden, activation_output, learning_rate=0.1, epochs=1000, batch_size=32, momentum=0.9, lambda_reg=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.lambda_reg = lambda_reg

        self.weights_input_hidden, self.biases_hidden, self.weights_hidden_output, self.biases_output = self.initialize_parameters()

        # TODO: decidere come gestire questa scelta delle attivazioni
        self.activation_hidden = self.relu if activation_hidden == "relu" else self.sigmoid
        self.activation_output = self.tanh if activation_output == "leaky_relu" else self.tanh
        self.activation_hidden_derivative = self.relu_derivative if activation_hidden == "relu" else self.sigmoid_derivative
        self.activation_output_derivative = self.tanh_derivative if activation_output == "leaky_relu" else self.tanh_derivative


    def initialize_parameters(self):
        np.random.seed(42)
        weights_input_hidden = np.random.uniform(low=-0.4, high=0.3, size=(self.input_size, self.hidden_size))
        biases_hidden = np.ones((1, self.hidden_size))
        weights_hidden_output = np.random.uniform(low=-0.4, high=0.3, size=(self.hidden_size, self.output_size))
        biases_output = np.ones((1, self.output_size))
        return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

    def tanh(self, x):
        return np.tanh(x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1.0, 0.0)
    
    def leaky_relu(self, x):
        return np.where(x >= 0, x, x * 0.01)
    
    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1.0, 0.01)

    def forward_propagation(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.biases_hidden
        hidden_layer_output = self.activation_hidden(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.biases_output
        output = self.activation_output(output_layer_input)

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

        output_delta = error_output * self.activation_output_derivative(output)
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.activation_hidden_derivative(hidden_layer_output)

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
    """
    def cross_validate(self, X, y, test_size=0.2):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        train_losses, val_losses, accuracies = [], [], []

        for epoch in range(self.epochs):
            indices = np.random.permutation(len(y_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, len(y_train), self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                hidden_layer_output, output = self.forward_propagation(X_batch)
                loss = self.calculate_loss(y_batch, output)
                self.backward_propagation(X_batch, y_batch, hidden_layer_output, output)

            train_losses.append(loss)

            # Evaluate on validation set
            val_loss = self.evaluate(X_val, y_val)
            val_losses.append(val_loss)

            # Compute accuracy on validation set
            val_predictions = self.predict(X_val)
            accuracy = self.compute_accuracy(y_val, val_predictions)
            accuracies.append(accuracy)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}, Accuracy: {accuracy}")

        # Plotting
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), train_losses, label='Training Loss')
        plt.plot(range(self.epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.savefig('learning_curve.png')  
        plt.close()

        plt.subplot(1, 2, 2)
        plt.plot(range(self.epochs), accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.savefig('Accuracy_validation.png')  
        plt.close()
        return val_losses[-1], accuracies[-1]
        
    """
    def predict(self, X):
        _, output = self.forward_propagation(X)
        return np.round(output)

    def compute_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    
    def evaluate(self, X, y):
        _, output = self.forward_propagation(X)
        loss = self.calculate_loss(y, output)
        return loss

"""
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

X_train, y_train = Preprocessing.preprocessing(percorso_file_train_1)

print("Monk-shape: ", X_train.shape)
print("Targets-shape: ", y_train.shape)

input_size = X_train.shape[1]

hidden_size = 3

nn = NeuralNetwork(input_size, hidden_size, learning_rate=0.9, epochs=450, batch_size=X_train.shape[0], momentum=0.9, lambda_reg=0.000000001)
nn.train(X_train, y_train)

# Cross-validate
#validation_loss, accuracy = nn.cross_validate(X_train, y_train)
#print(f"Final Validation Loss: {validation_loss}, Final Accuracy: {accuracy}")
"""