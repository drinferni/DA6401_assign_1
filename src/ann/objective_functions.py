import numpy as np
"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""


import numpy as np

def softmax(x):
    shift_x = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(shift_x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# def get_derivation_loss(y_true, y_pred, loss, activation, z_mat):
#     if loss == "mse":
#         if activation == "relu"
#         elif activation == "tanh"
#         elif activation == "sigmoid"

#     else:
#         y_pred = softmax(y_pred)
#         epsilon = 1e-15
#         y_pred = np.clip(y_pred, epsilon, 1.0)
#         return np.where(y_pred > 0, -y_true / y_pred, 0)

import numpy as np

def softmax(x):
    # Stabilized softmax
    shift_x = x - np.max(x, axis=0, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=0, keepdims=True)

def get_derivation_loss(y_true, y_pred, loss, activation, z_mat):
    """
    Returns dL/dz for the output layer.
    y_true: Ground truth (one-hot encoded)
    y_pred: Activation of the last layer (A_L)
    loss: 'mse' or 'cross_entropy'
    activation: 'relu', 'tanh', 'sigmoid', or 'softmax'
    z_mat: Pre-activation of the last layer (Z_L)
    """
    
    if loss == "mse":
        # dL/dA for MSE is (y_pred - y_true)
        da = (y_pred - y_true)
        
        # Multiply by dA/dZ (derivative of the activation function)
        # if activation == "relu":
        #     return da * (z_mat > 0).astype(float)
        # elif activation == "tanh":
        #     return da * (1 - np.tanh(z_mat)**2)
        # elif activation == "sigmoid":
        sig = 1 / (1 + np.exp(-z_mat))
        return da * (sig * (1 - sig))
        # else:
        #     return da # For linear/identity

    else: # Cross Entropy
        # For Cross-Entropy + Softmax, dL/dZ is simply (y_pred - y_true)
        # We assume y_pred is already the output of the softmax
        #if activation == "softmax":
        return (y_pred - y_true)
        
        # # If CE is used with Sigmoid (binary-like classification)
        # elif activation == "sigmoid":
        #     return (y_pred - y_true)
            
        # else:
        #     # General case: dL/dA * dA/dZ
        #     # This is numerically unstable, but provided for completeness
        #     epsilon = 1e-15
        #     y_pred = np.clip(y_pred, epsilon, 1.0)
        #     # For Categorical Cross-Entropy
        #     da = - (y_true / (y_pred + epsilon))
        #     return da * (z_mat > 0)
        
    
def get_loss(y_true, y_pred, loss):
    if loss == "mse":
        y_pred = softmax(y_pred)
        return np.mean(np.square(y_true - y_pred))
    
    else: # cross-entropy=
        y_pred = softmax(y_pred)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))