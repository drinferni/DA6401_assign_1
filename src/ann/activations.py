import numpy as np
"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
# """
# "relu","sigmoid","tanh"

def activate(z_mat,activation):
    if activation == "relu":
        return np.maximum(0, z_mat)
    elif activation == "tanh":
        return np.tanh(z_mat)
    else:
        return 1 / (1 + np.exp(-z_mat))
    
def get_derivative_activation(activation,z_mat):
    if activation == "relu":
        return (z_mat > 0).astype(float)
    elif activation == "tanh":
        return 1 - np.tanh(z_mat)**2
    else:
        s = 1 / (1 + np.exp(-z_mat))
        return s * (1 - s)
