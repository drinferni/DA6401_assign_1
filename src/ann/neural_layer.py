import numpy as np
from ann.activations import *
"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

def weight_initialsize(nm,hs,method,n,m):
    wm = []
    inp = n
    for x in range (0,nm):
        wght_mat = np.array((inp+1,hs))
        if method == "random":
            wght_mat = np.random.randn(inp+1,hs)
        elif method == "one":
            wght_mat = np.ones((inp+1,hs))
        else :
            limit = np.sqrt(6 / (inp+hs))
            wght_mat = np.random.uniform(-limit, limit, size=(inp+1,hs))
        inp = hs
        wm.append(wght_mat)
    
    final_mat = np.array((hs+1,m))
    if method == "random":
        final_mat = np.random.randn(hs+1,m)
    elif method == "one":
        final_mat = np.ones((hs+1,m))
    else :
        limit = np.sqrt(6 / (hs+m))
        final_mat = np.random.uniform(-limit, limit, size=(hs+1,m))
    
    wm.append(final_mat)

    return wm

def forward_layer(weight_mat,activation,input):
    # print(weight_mat.shape,"ok",input.shape)
    wt_transpose = weight_mat.T
    z_mat = wt_transpose @ input
    finval = activate(z_mat,activation)
    return z_mat,finval

def backward_layer(pre_wgt,pre_loss,z_mat,activation,ind):
    # print(pre_wgt.shape,pre_loss.shape,z_mat.shape)
    if pre_wgt.shape[1] != pre_loss.shape[0]:
        pre_loss = pre_loss[1:]
    return (pre_wgt @ pre_loss) * np.insert(get_derivative_activation(activation,z_mat),0,1)