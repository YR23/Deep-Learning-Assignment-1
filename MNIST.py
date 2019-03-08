from keras.datasets import mnist
import backward_propagation as BackProp
import forward_propagation as FowProp
import training as train
import Dropout as dropout

def main():
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(BackProp.relu_backward(5,-2))



if __name__ == "__main__":
    main()