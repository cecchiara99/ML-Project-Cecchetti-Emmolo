import numpy as np
import matplotlib.pyplot as plt
from functions import *
from utils import *

class NeuralNetwork:
    def __init__(self, input_size, output_size, activation_hidden, activation_output, hidden_size=2, learning_rate=0.1, epochs=1000, batch_size=32, momentum=0.9, lambda_reg=0.01, w_init_limit=[-0.1, 0.1]):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.learning_rate = learning_rate
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.momentum = momentum
        self.lambda_reg = lambda_reg
        self.w_init_limit = w_init_limit

        self.weights_input_hidden, self.biases_hidden, self.weights_hidden_output, self.biases_output = self.initialize_parameters()

        # Set activation functions
        act_funcs = {
            'sigmoid': sigmoid,
            'tanh': tanh,
            'relu': relu,
            'leaky_relu': leaky_relu,
            'identity': identity
        }

        act_funcs_derivatives = {
            'sigmoid': sigmoid_derivative,
            'tanh': tanh_derivative,
            'relu': relu_derivative,
            'leaky_relu': leaky_relu_derivative,
            'identity': identity_derivative
        }

        self.activation_hidden = act_funcs[activation_hidden] if (activation_hidden in act_funcs) else sigmoid
        self.activation_output = act_funcs[activation_output] if (activation_output in act_funcs) else sigmoid
        self.activation_hidden_derivative = act_funcs_derivatives[activation_hidden] if (activation_hidden in act_funcs_derivatives) else sigmoid_derivative
        self.activation_output_derivative = act_funcs_derivatives[activation_output] if (activation_output in act_funcs_derivatives) else sigmoid_derivative

    def initialize_parameters(self):
        np.random.seed(42)
        weights_input_hidden = np.random.uniform(low=self.w_init_limit[0], high=self.w_init_limit[1], size=(self.input_size, self.hidden_size))
        biases_hidden = np.ones((1, self.hidden_size))
        weights_hidden_output = np.random.uniform(low=self.w_init_limit[0], high=self.w_init_limit[1], size=(self.hidden_size, self.output_size))
        biases_output = np.ones((1, self.output_size))
        return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

    def forward_propagation(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.biases_hidden
        hidden_layer_output = self.activation_hidden(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.biases_output
        output = self.activation_output(output_layer_input)

        return hidden_layer_output, output

    def calculate_loss(self, y, y_pred):
        m = len(y)
        loss = np.mean((y_pred - y) ** 2)
        return loss

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
        self.weights_hidden_output -= self.momentum * self.learning_rate * np.clip(weights_hidden_output_gradient, -1e10, 1e10)
        self.biases_output -= self.momentum * self.learning_rate * np.clip(biases_output_gradient, -1e10, 1e10)
        self.weights_input_hidden -= self.momentum * self.learning_rate * np.clip(weights_input_hidden_gradient, -1e10, 1e10)
        self.biases_hidden -= self.momentum * self.learning_rate * np.clip(biases_hidden_gradient, -1e10, 1e10)

    def train_monk(self, X, y, X_test, y_test,task):
        m = len(y)
        losses = []
        test_losses = []
        
        accuracies = []
        accuracies_test = []

        for epoch in range(self.epochs):

            batch_loss = 0
            
            for i in range(0, m, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                
                
                hidden_layer_output, output = self.forward_propagation(X_batch)
                loss = self.calculate_loss(y_batch, output)
                batch_loss += loss
                    
                self.backward_propagation(X_batch, y_batch, hidden_layer_output, output)
            
            losses.append(loss)  
            
            accuracies.append(self.compute_accuracy(y, self.predict(X)))
            test_predictions = self.predict(X_test)
            test_losses.append(self.calculate_loss(y_test, test_predictions))
            accuracy_test = self.compute_accuracy(y_test, test_predictions)
            accuracies_test.append(accuracy_test)

        return losses, test_losses, accuracies, accuracies_test
  
    def train_cup(self, X, y, X_test, y_test,task):
        m = len(y)
        losses = []
        test_losses = []
        
        mees = []
        mees_test = []

        for epoch in range(self.epochs):

            batch_loss = 0
            
            for i in range(0, m, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                #X_batch = X_shuffled[i:i + self.batch_size]
                #y_batch = y_shuffled[i:i + self.batch_size]
                
                hidden_layer_output, output = self.forward_propagation(X_batch)
                    
                self.backward_propagation(X_batch, y_batch, hidden_layer_output, output)
            
            
            mees.append(mean_euclidean_error(y, self.predict(X)))
            
            test_predictions = self.predict(X_test)
            mees_test.append(mean_euclidean_error(y_test, test_predictions))
           

        return mees, mees_test 

    def predict(self, X):
        _, output = self.forward_propagation(X)
        return np.round(output)

    def compute_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def evaluate(self, X, y,task):
        _, output = self.forward_propagation(X)

        if task != "cup":
            loss = mean_squared_error(y, output)
        else:
            loss = mean_euclidean_error(y, output)
        return loss
