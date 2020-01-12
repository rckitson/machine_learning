""" Some sample functions for various tests

Test functions taken from Wikipedia:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
import numpy as np


def rosenbrock(x):
    """ Rosenbrock test function """
    assert len(x) > 1, "Must pass a vector to rosenbrock"

    value = 0
    for ii in range(len(x) - 1):
        value += 100 * (x[ii + 1] - x[ii] ** 2) ** 2 + (1 - x[ii]) ** 2
    return value


def circle(x):
    """ Circle test function """
    return np.sum(x ** 2, axis=1)


def rastrigin(x):
    """ Rastrigin function """
    A = 10
    return A * x.shape[1] + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=1)


def forrester_high_fidelity(x):
    """ The high-fidelity function from Forrester et al.

    doi:10.1098/rspa.2007.1900
    """
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def forrester_low_fidelity(x, y=None):
    """ The low-fidelity function from Forrester et al.

    doi:10.1098/rspa.2007.1900
    """
    if y is None:
        y = forrester_high_fidelity(x)
    A = np.ptp(y, axis=0) / 40.0
    B = np.ptp(y, axis=0) / 2.0
    C = -np.ptp(y, axis=0) / 4.0
    return A * y + B * (np.sum(x, axis=1) - 0.5) + C
