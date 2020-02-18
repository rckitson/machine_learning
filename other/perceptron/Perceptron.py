import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neuron(x, w, b):
    y = w*x + b
    return sigmoid(y)

x = np.linspace(-1,1,200)*10
# y = sigmoid(x)

plt.figure()
for w in range(1,2):
    for b in range(3):
        y = neuron(x, w, b)
        plt.plot(x, y, label='%d %d' % (w, b))
plt.legend()
plt.show()


