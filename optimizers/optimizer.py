#!/opt/anaconda3/bin/python3
""" Class definition for the optimizer 

Implemented based on the blog post here:
https://mmahesh.github.io/articles/2017-07/tutorial-on-sc-adagrad-a-new-stochastic-gradient-method-for-deep-learning

"""
import numpy as np


class Optimizer:
    """ A class for minimizing a function """

    def __init__(self, x0, function, method="sgd",
                 error_threshold=1e-4, learning_rate=0.1):
        """ Constructor 

        Args:
            x0: The initial point
            function: The function to optimize
            method: The optimization method 
        """
        self.x0 = x0
        self.function = function
        self.error_threshold = error_threshold

        # Hyperparameters needed for some of the algorithms
        self._m = 0
        self._v = 0
        self._update = np.random.random(len(x0)) * 1e-3

        if method == 'sgd':
            self.method = lambda x, g, t: self.sgd(x, g, learning_rate)
        elif method == 'momentum':
            self.method = lambda x, g, t: self.momentum(x, g, learning_rate)
        elif method == 'adam':
            self.method = lambda x, g, t: self.adam(x, g, learning_rate, t)
        elif method == 'rmsprop':
            self.method = lambda x, g, t: self.rms_prop(x, g, learning_rate)
        elif method == 'adagrad':
            self.method = lambda x, g, t: self.adagrad(x, g, learning_rate)
        elif method == 'adadelta':
            self.method = lambda x, g, t: self.adadelta(x, g)
        else:
            print("Unknown method")
        print("Using method: " + method)

    def solve(self):
        """ The main routine to solve the optimization problem """

        count = 0
        value, gradient = self.evaluate_function(self.x0)
        next_point = self.method(self.x0, gradient, count)

        output_file = open("history.dat", "w")
        output_file.write("{} {} {}\n".format(count + 1, value, np.linalg.norm(gradient)))

        value0 = value
        timeout = False
        while np.linalg.norm(gradient) > self.error_threshold:
            count += 1
            value, gradient = self.evaluate_function(next_point)
            next_point = self.method(next_point, gradient, count)
            output_file.write("{} {} {}\n".format(count + 1, value, np.linalg.norm(gradient)))
            if count % 1e3 == 0:
                print("{} {} {}".format(count, value, np.linalg.norm(gradient)))
            elif count > 5e4 or value > 10 * value0:
                timeout = True
                break

        if not timeout:
            print("Minimum reached")
        print("Current point")
        print("x: {}".format(next_point))
        print("f(x): {}".format(value))
        output_file.close()

    def evaluate_function(self, x):
        """ Return the function value and gradient 
        
        Args:
            x: The point to evaluate at
        
        Returns:
            A tuple: (function value, gradient) 
        """

        rand = np.random.random(len(x))
        delta = 1e-6 * rand / np.linalg.norm(rand)

        value = self.function(x)
        gradient = (self.function(x + delta) - value) / delta
        return value, gradient

    def sgd(self, x, gradient, learning_rate):
        """ Stochastic gradient descent 

        Args:
            x: The current position
            gradient: The current gradient at x
            learning_rate: The learning rate

        Returns:
            The new position
        """
        self._update = -learning_rate * gradient
        return x + self._update

    def momentum(self, x, gradient, learning_rate):
        """ Stochastic gradient descent with momentum

        Args:
            x: The current position
            gradient: The current gradient at x
            learning_rate: The learning rate

        Returns:
            The new position
        """
        rho = 0.9
        self._update = rho * self._update - learning_rate * gradient
        return x + self._update

    def adam(self, x, gradient, learning_rate, count):
        """ The Adam algorithm 

        https://arxiv.org/pdf/1412.6980.pdf

        Args:
            x: The current position
            gradient: The current gradient at x
            learning_rate: The learning rate
            count: The iteration count

        Returns:
            The new position
        """
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = count + 1

        self._m = beta1 * self._v + (1 - beta1) * gradient
        self._v = beta2 * self._v + (1 - beta2) * gradient ** 2

        m_hat = self._m / (1 - beta1 ** t)
        v_hat = self._v / (1 - beta2 ** t)

        self._update = -learning_rate * m_hat / (v_hat ** 0.5 + eps)
        return x + self._update

    def rms_prop(self, x, gradient, learning_rate):
        """ The RMS Prop algorithm 

        Args:
            x: The current position
            gradient: The current gradient at x
            learning_rate: The learning rate

        Returns:
            The new position
        """
        beta = 0.9
        eps = 1e-8

        self._v = beta * self._v + (1 - beta) * gradient ** 2

        self._update = -learning_rate * gradient / (self._v ** 0.5 + eps)
        return x + self._update

    def adagrad(self, x, gradient, learning_rate):
        """ The ADAGRAD algorithm 

        Args:
            x: The current position
            gradient: The current gradient at x
            learning_rate: The learning rate

        Returns:
            The new position
        """
        eps = 1e-8
        # Let rho be in [0,1] for momentum
        rho = 0.0

        self._v = self._v + gradient ** 2

        self._update = rho * self._update + -learning_rate * gradient / (self._v ** 0.5 + eps)
        return x + self._update

    def adadelta(self, x, gradient):
        """ The ADADELTA algorithm

        https://arxiv.org/pdf/1212.5701.pdf

        Args:
            x: The current position
            gradient: The current gradient at x

        Returns:
            The new position
        """
        # Best values for MNIST test (Table 2)
        eps = 1e-8
        rho = 0.95

        self._m = rho * self._m + (1 - rho) * gradient ** 2
        self._update = -rms(self._update) / (rms(gradient) + eps) * gradient
        self._v = rho * self._v + (1 - rho) * self._update ** 2

        return x + self._update


def rms(x):
    """ Root mean square 

    Args:
        x: numpy array
    """
    return np.linalg.norm(x) / np.sqrt(len(x))
