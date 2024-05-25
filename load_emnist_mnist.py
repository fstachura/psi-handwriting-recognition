import numpy as np
import os

def load_emnist_x(fn):
    with open(fn, "rb") as f:
        f.seek(16)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(-1, 28, 28).swapaxes(1, 2)

def load_emnist_y(fn):
    with open(fn, "rb") as f:
        f.seek(8)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data

def load_emnist_mnist(d):
    X_train = load_emnist_x(os.path.join(d, "emnist-mnist-train-images-idx3-ubyte"))
    X_test = load_emnist_x(os.path.join(d, "emnist-mnist-test-images-idx3-ubyte"))
    y_train = load_emnist_y(os.path.join(d, "emnist-mnist-train-labels-idx1-ubyte"))
    y_test = load_emnist_y(os.path.join(d, "emnist-mnist-test-labels-idx1-ubyte"))
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from utils import plot_mnist
    X_train, X_test, y_train, y_test = load_emnist_mnist("emnist")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    plot_mnist(X_train, y_train)
    plot_mnist(X_test, y_test)

