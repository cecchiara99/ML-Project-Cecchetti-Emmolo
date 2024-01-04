import numpy as np
import pandas as pd

class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid',
                 learning_rate=0.01, momentum=0.9, weight_decay=0.001):
        """
        Initialize the neural network layer with specified parameters.

        Args:
            input_size (int): Number of input neurons.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons.
            activation (str, optional): Activation function ('sigmoid' or 'relu').
            learning_rate (float, optional): Learning rate for gradient descent.
            momentum (float, optional): Momentum term for gradient descent.
            weight_decay (float, optional): Weight decay term for regularization.
        """

        # Initialize the layer with specified parameters
        self.input_size = input_size
        #self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.inputs = None
        self.outputs = None

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        # Initialize momentum terms
        self.momentum_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.momentum_bias_hidden = np.zeros_like(self.bias_hidden)
        #self.momentum_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        #self.momentum_bias_output = np.zeros_like(self.bias_output)

        # Set activation functions
        self.activation = self.sigmoid if activation == 'sigmoid' else self.relu
        self.output_activation = self.sigmoid

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x (numpy.ndarray): Input to the sigmoid function.

        Returns:
            numpy.ndarray: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid activation function.

        Args:
            x (numpy.ndarray): Output of the sigmoid function.

        Returns:
            numpy.ndarray: Derivative of the sigmoid function.
        """
        return x * (1 - x)

    def relu(self, x):
        """
        Rectified Linear Unit (ReLU) activation function.

        Args:
            x (numpy.ndarray): Input to the ReLU function.

        Returns:
            numpy.ndarray: Output of the ReLU function.
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        Derivative of the ReLU activation function.

        Args:
            x (numpy.ndarray): Output of the ReLU function.

        Returns:
            numpy.ndarray: Derivative of the ReLU function.
        """
        return np.where(x > 0, 1, 0)
