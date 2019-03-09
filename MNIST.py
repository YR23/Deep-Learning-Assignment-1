from keras.datasets import mnist
import backward_propagation as BackProp
import forward_propagation as FowProp
import training as train
from keras.utils import to_categorical
import Dropout as dropout
import numpy as np
import math

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_train = x_train/255
    x_train = x_train.T
    print(y_train.shape)
    y_train = to_categorical(y_train).T
    print(y_train.shape)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    layers = [x_test.shape[1],20,7,5,10]
    train.L_layer_model(x_train,y_train,layers,0.0009,1000,32)


if __name__ == "__main__":
    main()