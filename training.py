import numpy as np
import backward_propagation as BackProp
import forward_propagation as FowProp
import math
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import Softmax

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
def SplitVal(X, Y):
    return train_test_split(X.T, Y.T, test_size=0.2, shuffle=True)


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    costs = []
    x_train,x_val,y_train,y_val = SplitVal(X,Y)
    parameters = FowProp.initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        batches_X = CreateBatches(x_train.T,batch_size)
        batches_Y = CreateBatches(y_train.T,batch_size)
        for j in range(len(batches_X)):
            AL, caches = FowProp.L_model_forward(batches_X[j], parameters)
            cost = FowProp.compute_cost(AL, batches_Y[j])
            grads = BackProp.L_model_backward(AL, batches_Y[j], caches)
            parameters = BackProp.Update_parameters(parameters, grads, learning_rate)
        print("cost for iteration "+str(i)+": "+str(cost))
        Predict(x_val.T,y_val.T,parameters)
        if i % 100 == 0:
            costs.append(cost)
    return parameters,costs


# b.
def Predict(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros(m)
    probas, caches = FowProp.L_model_forward(X, parameters)
    #probas = Softmax(probas)
    correct = 0
    for i in range(m):
        indexX = -1
        indexY = -1
        maxX = -1
        for j in range(10):
            if probas[j][i] > maxX:
                maxX = probas[j][i]
                indexX = j
            if (Y[j][i] == 1):
                indexY = j
        if indexX == indexY:
            correct += 1
    return correct / m

def softmax(x):
    return (x/ np.sum(x,axis=0))