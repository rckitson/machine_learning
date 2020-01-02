#!/opt/anaconda3/bin/python3
""" Tests for the Optimizer class

Test functions taken from Wikipedia:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
import os
import glob
import subprocess
import shutil
import numpy as np
from matplotlib import pyplot as plt
import optimizer

np.random.seed(0)
for ff in glob.glob('*.dat'):
    os.remove(ff)

TOLERANCE = 1e-3
LEARNING_RATE = 5e-3
# See the first few methods for available test functions
TEST_FUNCTION = lambda x: circle(x)
# This can be increased to higher dimensions
X0 = np.random.random(1)
X0 = X0/np.linalg.norm(X0) 
print(X0)

def rosenbrock(x):
    """ Rosenbrock test function """
    assert len(x) > 1, "Must pass a vector to rosenbrock"

    value = 0
    for ii in range(len(x) - 1):
        value += 100*(x[ii+1] - x[ii]**2)**2 + (1 - x[ii])**2
    return value

def circle(x):
    """ Circle test function """
    return np.sum(x**2)
    
def test_all():
    for algo in ['sgd', 'momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta']: 
        opt = optimizer.Optimizer(x0=X0, function=TEST_FUNCTION, method=algo,
                learning_rate=LEARNING_RATE, error_threshold=TOLERANCE)
        opt.solve()
        shutil.move('history.dat', algo + '.dat')

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
    

if __name__=="__main__":
    test_all()
    plot_history()
