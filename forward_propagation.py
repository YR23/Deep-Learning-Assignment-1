import numpy as np

"""
The functions of the first part: Forward propagation process
"""


# a.
def initialize_parameters(layer_dims):
    dic = {}
    for i in range (1,len(layer_dims)):
        dic['W'+str(i)] = np.random.random((layer_dims[i],layer_dims[i-1]))
        dic['b'+str(i)] = np.zeros((layer_dims[i],1))
    return dic


# b.
def linear_forward(A, W, b):
    return 1


# c.
def sigmoid(Z):
    return 1


# d.
def relu(Z):
    return 1


# e.
def linear_activation_forward(A_prev, W, B, activation):
    return 1


# f.
def L_model_forward(X, parameters, use_batchnorm):
    return 1


# g.
def compute_cost(AL, Y):
    return 1


# h.
def apply_batchnorm(A):
    return

