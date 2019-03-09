from keras.datasets import mnist
import backward_propagation as BackProp
import forward_propagation as FowProp
import training as train
from keras.utils import to_categorical
import Dropout as dropout
import numpy as np
import math
import time

def main():
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_train = x_train/255
    x_train = x_train.T
    y_train = to_categorical(y_train).T
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    layers = [x_test.shape[1],20,10]
    train.L_layer_model(x_train,y_train,layers,0.0009,1000,32)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()