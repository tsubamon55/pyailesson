import numpy as np
import matplotlib.pyplot as plt
import activation_functions as acfunc

inp_a = np.arange(-1.0, 1.0, 0.2)
inp_b = np.arange(-1.0, 1.0, 0.2)

result = np.zeros((10, 10))

w_mid = np.array([[4.0, 4.0],
                 [4.0, 4.0],
                  ])

w_out = np.array([[1.0],
                 [-1.0],
                  ])

b_mid = np.array([3.0, -3.0])
b_out = np.array([0.1])


def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return acfunc.sigmoid_func(u)


def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return u


if __name__ == '__main__':
    for i in range(10):
        for j in range(10):
            inp = np.array(inp_a[i], inp_b[j])
            mid = middle_layer(inp, w_mid, b_mid)
            out = output_layer(mid, w_out, b_out)

            result[j][i] = out[0]

    plt.imshow(result, "gray", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.show()
