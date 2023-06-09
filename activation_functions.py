import numpy as np
import matplotlib.pylab as plt


def sigmoid_func(x):
    return 1/(1+np.exp(-x))


def tanh_func(x):
    return np.tanh(x)


def relu_func(x):
    return np.where(x <= 0, 0, x)


def leaky_relu_func(x):
    return np.where(x <= 0, 0.01*x, x)


def softmax_func(x):
    return np.exp(x)/np.sum(np.exp(x))


if __name__ == '__main__':
    x = np.linspace(-5, 5)
    y = softmax_func(x)
    plt.plot(x, y)
    plt.show()
