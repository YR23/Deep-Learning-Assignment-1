from keras.datasets import mnist
import backward_propagation as BackProp
import forward_propagation as FowProp
import training as train
import Dropout as dropout
import numpy as np
import math

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    layers = [x_test.shape[1],20,7,5,10]
    print("Cheking initalization")
    print("W shape should be (current layer, prev layer)")
    print("b shape should be (current layer, 1)")
    for i in range (len(layers)):
        if i != len(layers)-1:
            print("W"+str(i+1) + " shape: "+str(FowProp.initialize_parameters(layers)["W"+str(i+1)].shape))
            print("b"+str(i+1) + " shape: "+str(FowProp.initialize_parameters(layers)["b"+str(i+1)].shape))
    train.L_layer_model(x_train,y_train,layers,0.0009,1000,32)


if __name__ == "__main__":
    main()