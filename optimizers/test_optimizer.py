#!/opt/anaconda3/bin/python3
""" Tests for the Optimizer class

Test functions taken from Wikipedia:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
import os
import glob
import shutil
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import optimizer


def main():
    """ The main routine """
    np.random.seed(0)
    for ff in glob.glob('*.dat'):
        os.remove(ff)

    tolerance = 1e-6
    learning_rate = 1e-4
    function = circle
    # See the first few methods for available test functions
    # x0 can be increased to higher dimensions
    x0 = np.random.random(3)
    x0 = x0 / np.linalg.norm(x0)

    test_all(x0, function, learning_rate, tolerance)
    plot_history()
    subprocess.call(['open', 'convergence_history.png'])


def rosenbrock(x):
    """ Rosenbrock test function """
    assert len(x) > 1, "Must pass a vector to rosenbrock"

    value = 0
    for ii in range(len(x) - 1):
        value += 100 * (x[ii + 1] - x[ii] ** 2) ** 2 + (1 - x[ii]) ** 2
    return value


def circle(x):
    """ Circle test function """
    return np.sum(x ** 2)


def test_all(x0, function, learning_rate, tolerance):
    """ Test all the algorithms

    Args:
        x0: The initial point
        function: The test function
        learning_rate: The learning rate
        tolerance: The tolerance on the gradient

    """
    for algorithm in ['sgd', 'momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta']:
        opt = optimizer.Optimizer(x0=x0, function=function, method=algorithm,
                                  learning_rate=learning_rate, error_threshold=tolerance)
        opt.solve()
        shutil.move('history.dat', algorithm + '.dat')


def plot_history():
    files = glob.glob('*.dat')

    plt.figure()
    for ff in files:
        if ff == 'history.dat':
            continue

        dat = np.loadtxt(ff)
        if len(dat) > 1000:
            skip = max(1, len(dat) // 10)
        else:
            skip = 1
        plt.plot(dat[::skip, 0], dat[::skip, 1], label=os.path.splitext(ff)[0])

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Residual')
    plt.savefig('convergence_history.png')


if __name__ == "__main__":
    main()
