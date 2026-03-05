import numpy as np

# class SGD:
#     def __init__(self, lr=0.01):
#         self.lr = lr

#     def preprocess(self, W):
#         return W

#     def update(self, W, G,d):
#         for i in range(len(W)):
#             W[i] -= self.lr * G[i]
#         return W

# class Momentum:
#     def __init__(self, lr=0.01, beta=0.9):
#         self.lr = lr
#         self.beta = beta
#         self.v = None

#     def preprocess(self, W):
#         return W

#     def update(self, W, G,d):
#         if self.v is None:
#             self.v = [np.zeros_like(w) for w in W]
            
#         for i in range(len(W)):
#             self.v[i] = self.beta * self.v[i] + self.lr * G[i]
#             W[i] -= self.v[i]
#         return W

# class NAG:
#     """Nesterov Accelerated Gradient"""
#     def __init__(self, lr=0.01, beta=0.9):
#         self.lr = lr
#         self.beta = beta
#         self.v = None

#     def preprocess(self, W):
#         """Move weights in the direction of momentum before forward pass"""
#         if self.v is None:
#             self.v = [np.zeros_like(w) for w in W]
        
#         for i in range(len(W)):
#             W[i] -= self.beta * self.v[i]
#         return W

#     def update(self, W, G,d):
#         # W is already 'shifted' from preprocess. 
#         # We update velocity and then correct the weight position.
#         for i in range(len(W)):
#             v_prev = self.v[i].copy()
#             self.v[i] = self.beta * self.v[i] + self.lr * G[i]
#             # Standard NAG: W_new = W_old - beta*v_old - (v_new)
#             # Since preprocess already did W - beta*v_old, we just subtract the new velocity
#             W[i] -= self.v[i] 
#         return W

# class RMSprop:
#     def __init__(self, lr=0.01, beta=0.99, eps=1e-8):
#         self.lr = lr
#         self.beta = beta
#         self.eps = eps
#         self.s = None

#     def preprocess(self, W):
#         return W

#     def update(self, W, G,d):
#         if self.s is None:
#             self.s = [np.zeros_like(w) for w in W]
            
#         for i in range(len(W)):
#             self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (G[i]**2)
#             W[i] -= self.lr * G[i] / (np.sqrt(self.s[i]) + self.eps)
#         return W


## manually check this before sleeping


class SGD:
    def __init__(self,weight_decay=1e-4, lr=0.01 ):
        self.lr = lr
        self.weight_decay = weight_decay

    def preprocess(self, W):
        return W

    def update(self, W, G, d):
        for i in range(len(W)):
            # 1. Apply Weight Decay to weights (row 1 onwards)
            # Formula: W = W - lr * weight_decay * W
            W[i][1:] -= self.lr * self.weight_decay * W[i][1:]
            
            # 2. Apply Gradient Update to all (including bias)
            W[i] -= self.lr * G[i]
        return W
    

import numpy as np

class Momentum:
    def __init__(self, weight_decay=1e-4, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = None

    def preprocess(self, W):
        return W

    def update(self, W, G, d):
        if self.v is None:
            self.v = [np.zeros_like(w) for w in W]

        for i in range(len(W)):
            # Create a copy of gradient to add decay without mutating original G
            grad_with_decay = G[i].copy()
            # Add L2 penalty to weights only (row 1 onwards)
            grad_with_decay[1:] += self.weight_decay * W[i][1:]
            
            # Velocity update
            self.v[i] = self.momentum * self.v[i] - self.lr * grad_with_decay
            W[i] += self.v[i]
        return W
    
class NAG:
    def __init__(self,weight_decay=1e-4, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = None

    def preprocess(self, W):
        if self.v is None:
            return W
        # Lookahead: W_temp = W + momentum * v
        return [W[i] + self.momentum * self.v[i] for i in range(len(W))]

    def update(self, W, G, d):
        if self.v is None:
            self.v = [np.zeros_like(w) for w in W]

        for i in range(len(W)):
            # G was calculated at the lookahead position in your script logic
            grad_with_decay = G[i].copy()
            grad_with_decay[1:] += self.weight_decay * W[i][1:]
            
            prev_v = self.v[i].copy()
            self.v[i] = self.momentum * self.v[i] - self.lr * grad_with_decay
            
            # Update the original weights
            W[i] += -self.momentum * prev_v + (1 + self.momentum) * self.v[i]
        return W
    
class RMSprop:
    def __init__(self, lr=0.01, alpha=0.9, eps=1e-8, weight_decay=1e-4):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.s = None

    def update(self, W, G, d):
        if self.s is None:
            self.s = [np.zeros_like(w) for w in W]

        for i in range(len(W)):
            # 1. Update squared gradient cache (standard RMSProp)
            self.s[i] = self.alpha * self.s[i] + (1 - self.alpha) * (G[i]**2)
            
            # 2. Step 1: Apply Decoupled Weight Decay to weights only (Row 1+)
            # This is applied independently of the adaptive learning rate
            W[i][1:] -= self.lr * self.weight_decay * W[i][1:]
            
            # 3. Step 2: Apply the standard RMSProp gradient update
            W[i] -= self.lr * G[i] / (np.sqrt(self.s[i]) + self.eps)
        return W