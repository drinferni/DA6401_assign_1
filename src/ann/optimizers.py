import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def preprocess(self, W):
        return W

    def update(self, W, G):
        for i in range(len(W)):
            W[i] -= self.lr * G[i]
        return W

class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def preprocess(self, W):
        return W

    def update(self, W, G):
        if self.v is None:
            self.v = [np.zeros_like(w) for w in W]
            
        for i in range(len(W)):
            self.v[i] = self.beta * self.v[i] + self.lr * G[i]
            W[i] -= self.v[i]
        return W

class NAG:
    """Nesterov Accelerated Gradient"""
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def preprocess(self, W):
        """Move weights in the direction of momentum before forward pass"""
        if self.v is None:
            self.v = [np.zeros_like(w) for w in W]
        
        for i in range(len(W)):
            W[i] -= self.beta * self.v[i]
        return W

    def update(self, W, G):
        # W is already 'shifted' from preprocess. 
        # We update velocity and then correct the weight position.
        for i in range(len(W)):
            v_prev = self.v[i].copy()
            self.v[i] = self.beta * self.v[i] + self.lr * G[i]
            # Standard NAG: W_new = W_old - beta*v_old - (v_new)
            # Since preprocess already did W - beta*v_old, we just subtract the new velocity
            W[i] -= self.v[i] 
        return W

class RMSprop:
    def __init__(self, lr=0.01, beta=0.99, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = None

    def preprocess(self, W):
        return W

    def update(self, W, G):
        if self.s is None:
            self.s = [np.zeros_like(w) for w in W]
            
        for i in range(len(W)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (G[i]**2)
            W[i] -= self.lr * G[i] / (np.sqrt(self.s[i]) + self.eps)
        return W
