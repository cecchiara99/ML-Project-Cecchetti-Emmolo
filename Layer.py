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
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.inputs = None
        self.outputs = None
        self.delta = None

        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size) # Formula di inizializzazione di He.
        self.bias = np.zeros((1, output_size))

        # Initialize momentum terms
        self.momentum_weights = np.zeros_like(self.weights)
        self.momentum_bias = np.zeros_like(self.bias)

        # Set activation functions
        act_funcs = {
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'relu': self.relu
        }

        act_funcs_derivatives = {
            'sigmoid': self.sigmoid_derivative,
            'tanh': self.tanh_derivative,
            'relu': self.relu_derivative
        }

        self.activation = act_funcs[activation]
        self.activation_derivative = act_funcs_derivatives[activation]
        # 

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
    
    def tanh(x):
        """
        Computes the hyperbolic tangent function (TanH).

        Args:
            x (numpy.ndarray): Input to the hyperbolic tangent function.

        Returns:
            numpy.ndarray: Output of the hyperbolic tangent function.
        """
        return np.tanh(x)

    def tanh_derivative(x):
        """
        Computes the derivative of the hyperbolic tangent function (TanH).

        Args:
            x (numpy.ndarray): Output of the hyperbolic tangent function.

        Returns:
            numpy.ndarray: Derivative of the hyperbolic tangent function.
        """
        return 1 - np.tanh(x) ** 2

    def forward_pass(self, inputs: np.ndarray):
        """
        Compute the forward pass of the layer.

        Args:
            inputs (numpy.ndarray): Inputs to the layer.

        Returns:
            numpy.ndarray: Outputs from the layer.
        """

        self.inputs = inputs
        """
        print("Inputs shape:", inputs.shape)
        print("Weights shape:", self.weights.shape)
        print("Bias shape:", self.bias.shape)
        """
        linear_output = np.dot(inputs, self.weights) + self.bias # Multiply by weights and add bias
        self.outputs = self.activation(linear_output) # Apply activation function

        threshold = 0.5
        binary_predictions = (self.outputs > threshold).astype(int)
        print("Binary predictions:", binary_predictions)
        
        """
            Gli output sembrano essere valori compresi tra 0 e 1, che è comune quando
                si utilizzano funzioni di attivazione come la sigmoide.
            Tuttavia, per interpretare meglio l'output, potresti voler arrotondarlo o
                utilizzare una soglia per convertire i valori in previsioni binarie,
                soprattutto se stai affrontando un problema di classificazione binaria.

            Ad esempio, se vuoi convertire i valori dell'output in previsioni binarie basate
                su una soglia, potresti fare qualcosa del genere:
            ```python
            threshold = 0.5
            binary_predictions = (output > threshold).astype(int)
            print("Binary predictions:", binary_predictions)
            ```
            In questo esempio, i valori superiori a 0.5 vengono convertiti in 1,
                mentre quelli inferiori o uguali a 0.5 vengono convertiti in 0.
                Tuttavia, la scelta della soglia dipende dal tuo problema specifico
                e dalle tue esigenze.
        """
        return self.outputs
    
    def backward_pass(self, output_gradient, learning_rate, weight_decay):
        """
        Backpropagate the gradient through the layer.

        Args:
            output_gradient (numpy.ndarray): Gradient of the error with respect to the output.
            learning_rate (float): Learning rate for gradient descent.

        Returns:
            numpy.ndarray: Gradient of the error with respect to the input.
        """

        # Compute the gradient respect to the inputs
        delta_inputs = np.dot(output_gradient, self.weights.T)

        # Compute the gradient respect to the weights and bias
        delta_weights = np.dot(self.inputs.T, output_gradient) + weight_decay * self.weights
        delta_bias = np.sum(output_gradient, axis=0, keepdims=True)

        # Update the weights and bias with momentum
        self.momentum_weights = self.momentum * self.momentum_weights + learning_rate * delta_weights
        self.momentum_bias = self.momentum * self.momentum_bias + learning_rate * delta_bias

        # Update the weights and bias
        self.weights += self.momentum_weights
        self.bias += self.momentum_bias

        # Restituisci il gradiente rispetto agli input per l'uso nelle retropropagazioni successive
        return delta_inputs
