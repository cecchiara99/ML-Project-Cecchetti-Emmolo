import numpy as np
import matplotlib.pyplot as plt
from functions import *
from utils import *

class NeuralNetwork:
    def __init__(self, input_size, output_size, activation_hidden, activation_output, hidden_size=2, learning_rate=0.1, epochs=1000, batch_size=32, momentum=0.9, lambda_reg=0.01, w_init_limit=[-0.1, 0.1]):
        """
        Initialize the neural network.

        :param input_size: the size of the input layer
        :param output_size: the size of the output layer
        :param activation_hidden: the activation function of the hidden layer
        :param activation_output: the activation function of the output layer
        :param hidden_size: the size of the hidden layer
        :param learning_rate: the learning rate
        :param epochs: the number of epochs
        :param batch_size: the batch size
        :param momentum: the momentum
        :param lambda_reg: the regularization parameter
        :param w_init_limit: the range of the initial weights
        """
        
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.learning_rate = learning_rate
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.momentum = momentum
        self.lambda_reg = lambda_reg
        self.w_init_limit = w_init_limit

        # Initialize weights and biases
        self.weights_input_hidden, self.biases_hidden, self.weights_hidden_output, self.biases_output = self.initialize_parameters()

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

        # Set activation functions
        self.activation_hidden = act_funcs[activation_hidden] if (activation_hidden in act_funcs) else sigmoid
        self.activation_output = act_funcs[activation_output] if (activation_output in act_funcs) else sigmoid
        self.activation_hidden_derivative = act_funcs_derivatives[activation_hidden] if (activation_hidden in act_funcs_derivatives) else sigmoid_derivative
        self.activation_output_derivative = act_funcs_derivatives[activation_output] if (activation_output in act_funcs_derivatives) else sigmoid_derivative


    def initialize_parameters(self):
        """
        Initialize the parameters of the neural network.

        :return: the initialized parameters
        """

        np.random.seed(42)
        weights_input_hidden = np.random.uniform(low=self.w_init_limit[0], high=self.w_init_limit[1], size=(self.input_size, self.hidden_size))
        biases_hidden = np.ones((1, self.hidden_size))
        weights_hidden_output = np.random.uniform(low=self.w_init_limit[0], high=self.w_init_limit[1], size=(self.hidden_size, self.output_size))
        biases_output = np.ones((1, self.output_size))

        return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output


    def forward_propagation(self, X):
        """
        Execute the forward propagation step of the neural network.

        :param X: the input data

        :return: the output of the hidden layer and the output of the output layer
        """

        # Compute the output of the hidden layer
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.biases_hidden
        hidden_layer_output = self.activation_hidden(hidden_layer_input)

        # Compute the output of the output layer
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.biases_output
        output = self.activation_output(output_layer_input)

        return hidden_layer_output, output
    

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


    def train(self, training_X, training_y, validation_X, validation_y, task, patience=10):
        m = len(training_y)
        losses = []
        best_validation_error = 100

        for epoch in range(self.epochs):
            loss = 0

            for i in range(0, m, self.batch_size):
                X_batch = training_X[i:i + self.batch_size]
                y_batch = training_y[i:i + self.batch_size]
                
                hidden_layer_output, output = self.forward_propagation(X_batch)
                batch_loss = mean_squared_error(y_batch, output)
                loss += batch_loss
                self.backward_propagation(X_batch, y_batch, hidden_layer_output, output)
            
            if task == 'cup':
                # Evaluate the model on the validation set
                validation_error = self.evaluate(validation_X, validation_y, task)
                # Check for early stopping
                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                    consecutive_no_improvement = 0
                else:
                    consecutive_no_improvement += 1

                if consecutive_no_improvement > patience:
                    print(f"Early stopping at epoch {epoch} with validation error {validation_error}")
                    n_epochs = epoch
                    break

            loss = loss / (m / self.batch_size)
            # Print the loss every 100 epochs
            #if epoch % 100 == 0:
            #    print(f"Epoch {epoch} - Loss: {loss} - Validation error: {validation_error}")
            

    def retrain(self, data_X, data_y, test_X, test_y, task='monk', patience=10):
        m = len(data_y)
        n_epochs = self.epochs
        best_validation_error = float('inf')
        consecutive_no_improvement = 0
        # For MONK
        losses = []
        test_losses = []
        accuracies = []
        test_accuracies = []
        # For CUP
        mees = []
        mees_test = []
        

        for epoch in range(self.epochs):
            loss = 0
            test_loss = 0
            
            for i in range(0, m, self.batch_size):
                X_batch = data_X[i:i + self.batch_size]
                y_batch = data_y[i:i + self.batch_size]
                
                hidden_layer_output, output = self.forward_propagation(X_batch)
                batch_loss = mean_squared_error(y_batch, output)
                loss += batch_loss
                batch_loss
                batch_test_loss = mean_squared_error(test_y, self.predict(test_X))
                test_loss += batch_test_loss
                self.backward_propagation(X_batch, y_batch, hidden_layer_output, output)
            
            if task == 'cup':
                # Evaluate the model on the validation set
                validation_error = self.evaluate(data_X, data_y, task)
                # Check for early stopping
                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                    consecutive_no_improvement = 0
                else:
                    consecutive_no_improvement += 1

                if consecutive_no_improvement > patience:
                    print(f"Early stopping at epoch {epoch} with validation error {validation_error}")
                    n_epochs = epoch
                    break

            loss = loss / (m/self.batch_size)
            test_loss = test_loss / (m/self.batch_size)
            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")
            
            if task != 'cup':
                losses.append(loss)
                test_losses.append(test_loss)
                accuracies.append(self.compute_accuracy(data_y, self.predict(data_X)))
                test_accuracies.append(self.compute_accuracy(test_y, self.predict(test_X)))
            else:
                mees.append(self.evaluate(data_X, data_y, task))
                mees_test.append(self.evaluate(test_X, test_y, task))
            
        if task != 'cup':
            print(f"Final loss: {losses[-1]}")
            print(f"Final test loss: {test_losses[-1]}")
            return losses, test_losses, accuracies, test_accuracies, n_epochs
        else:
            print(f"Final MEE: {mees[-1]}")
            print(f"Final test MEE: {mees_test[-1]}")
            return mees, mees_test, None, None, n_epochs


    def predict(self, X):
        _, output = self.forward_propagation(X)
        return np.round(output) # Round the output to 0 or 1 for classification, leave it as it is for regression


    def compute_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    

    def evaluate(self, X, y, task):
        output = self.predict(X)
        loss = 0
        if task != 'cup':
            loss = mean_squared_error(y, output)
        else:
            loss = mean_euclidean_error(y, output)
        return loss
