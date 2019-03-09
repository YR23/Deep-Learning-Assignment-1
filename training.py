import numpy as np
import backward_propagation as BackProp
import forward_propagation as FowProp
import math

"""
The functions of the Third part: Train and Predict
"""

# helping functions
def CreateBatches(X, batch_size):
    dev = math.ceil(X.shape[1] / batch_size)
    batches = []
    for i in range(dev):
        batches.append(X[:, i * batch_size:(i + 1) * batch_size])
    return batches


# a.
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    costs = []
    parameters = FowProp.initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        batches_X = CreateBatches(X,batch_size)
        batches_Y = CreateBatches(Y,batch_size)
        for j in range(len(batches_X)):
            AL, caches = FowProp.L_model_forward(batches_X[j], parameters)
            cost = FowProp.compute_cost(AL, batches_Y[j])
            grads = BackProp.L_model_backward(AL, batches_Y[j], caches)
            parameters = BackProp.Update_parameters(parameters, grads, learning_rate)
        print("cost for iteration "+str(i)+": "+str(cost))
        if i % 100 == 0:
            costs.append(cost)
    return parameters,costs


# b.
def Predict(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros(m)

    probas, caches = FowProp.L_model_forward(X, parameters)
    correct = 0
    for i in range(0, probas.shape[1]):
        max = -1
        for j in range (0,10):
            if probas[j, i] > max:
                p[i] = j
                max = probas[j, i]
        if p[i] == Y[i]:
            correct += 1
    return correct / m

