import numpy as np
import time
"""
The functions of the first part: Forward propagation process
"""


# a.
def initialize_parameters(layer_dims):
    dic = {}
    np.random.seed(int(time.time()))
    for i in range(1, len(layer_dims)):
        dic['W' + str(i)] = np.random.random((layer_dims[i], layer_dims[i - 1]))/100
        dic['b' + str(i)] = np.zeros((layer_dims[i], 1))
    return dic


# b.
def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache


# c.
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)),Z


# d.
def relu(Z):
    zeros = np.zeros((Z.shape[0], Z.shape[1]))
    rel = np.maximum(zeros, Z)
    return rel, Z


# e.
def linear_activation_forward(A_prev, W, B, activation):
    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == "sigmoid":
       A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    return A, (linear_cache, activation_cache)


# f.
def L_model_forward(X, parameters, use_batchnorm=False, dropout=0.5):
    caches = []
    N_layers = len(parameters) // 2
    A = X
    for l in range(1, N_layers):
        A_last = A
        A, cache = linear_activation_forward(A_last, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        A = DropOut(A, dropout)
        if (use_batchnorm):
            A =  apply_batchnorm(A)
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(N_layers)], parameters['b' + str(N_layers)], 'sigmoid')
    caches.append(cache)
    return AL, caches


# g.
def compute_cost(AL, Y):
    cost = (-1 / Y.shape[1]) * np.sum(np.multiply(np.log(AL),Y) + np.multiply( np.log(1 - AL),1-Y))
    if cost.shape != ():
        cost = np.squeeze(cost)
    return cost


# h.
def apply_batchnorm(A):
    return (A-np.mean(A, axis=0)) / (np.var(A, axis=0)+1e-8)

def DropOut(L,rate):
    drop = np.random.rand(L.shape[0],L.shape[1])
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if drop[i][j] <= rate:
                drop[i][j] = 0
            else:
                drop[i][j] = 1
    res = drop*L
    return res
