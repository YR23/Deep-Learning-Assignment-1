import numpy as np
import forward_propagation as FowProp


"""
The functions of the second part: Backward propagation process
"""


# a.
def Linear_backward(dZ, cache):
    A_prev, W, _ = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


# b.
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = Linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


# c.
def relu_backward(dA, activation_cache):
    return dA if (activation_cache > 0) else 0


# d.
def sigmoid_backward(dA, activation_cache):
    return dA * (FowProp.sigmoid(activation_cache)*(1-FowProp.sigmoid(activation_cache)))


# e.
def L_model_backward(AL, Y, caches):
    grads = {}
    num_layers = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[num_layers - 1]
    grads["dA" + str(num_layers)], grads["dW" + str(num_layers)], grads["db" + str(num_layers)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    for l in reversed(range(num_layers - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# f.
def Update_parameters(parameters, grads, learning_rate):
    for l in range(len(parameters) // 2):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters