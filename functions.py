import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
    
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu(x):
    return np.where(x >= 0, x, x * 0.01)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1.0, 0.01)

def identity(x):
    return x

def identity_derivative(x):
    return 1