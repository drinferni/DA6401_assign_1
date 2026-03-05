import numpy as np
"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""


import numpy as np

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def get_derivation_loss(y_true, y_pred, loss):
    if loss == "mse":
        return y_pred - y_true

    else:
        y_pred = softmax(y_pred)
        return np.where(y_pred > 0, -y_true / y_pred, 0)
    
def get_loss(y_true, y_pred, loss):
    if loss == "mse":
        return np.mean(np.square(y_true - y_pred))
    
    else: # cross-entropy=
        y_pred = softmax(y_pred)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))