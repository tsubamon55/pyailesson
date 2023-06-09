import numpy as np
import matplotlib.pyplot as plt
import activation_functions as acfunc

inp_a = np.arange(-1.0, 1.0, 0.2)
inp_b = np.arange(-1.0, 1.0, 0.2)

outputs = np.zeros((10, 10))

weight_a = 2.5
weight_b = 3

bias = 0.1

for i in range(10):
    for j in range(10):
        u_single = inp_a[i] * weight_a + inp_b[j] * weight_b + bias
        outputs[i][j] = acfunc.sigmoid_func(u_single)

plt.imshow(outputs, "gray", vmin=0.0, vmax=1.0)
plt.colorbar()
plt.show()
