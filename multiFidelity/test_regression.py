#!/opt/anaconda3/bin/python3
""" Some tests of the kriging and cokriging class

This assumes the PYTHONPATH includes the machine_learning directory
"""
import numpy as np
from matplotlib import pyplot as plt
import regression
import test_functions

np.random.seed(0)


def test_rbf():
    rbf = regression.RadialBasisFunction()
    test_function = test_functions.rastrigin

    N = 2 ** np.arange(1, 6)
    error = np.zeros(len(N))
    for ii in range(len(error)):
        x_train = np.linspace(-1, 1, N[ii]) * np.pi * 2
        x_train = x_train.reshape(-1, 1)
        y_train = test_function(x_train)
        y_train = y_train.reshape(-1, 1)
        rbf.fit(x_train, y_train)

        x_test = np.linspace(-1, 1, 10 * len(x_train)) * np.pi * 2
        x_test = x_test.reshape(-1, 1)
        y_test = test_function(x_test)
        y_predict = rbf.predict(x_test)
        print(y_test - y_predict)
        error[ii] = np.mean((y_predict - y_test) ** 2)

        if ii == len(error) - 1:
            plt.figure()
            plt.scatter(x_train, y_train)
            plt.plot(x_test, y_test)
            plt.plot(x_test, y_predict, label='RBF')
            plt.legend()
            plt.show()

    plt.figure()
    plt.plot(N, error, '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel('Training points')
    plt.show()


def test_kriging():
    kriging = regression.Kriging(scale=False)
    test_function = test_functions.rastrigin
    x_range = 5.12

    N = 2 ** np.arange(1, 8)
    error = np.zeros(len(N))
    for ii in range(len(error)):
        train_x = np.linspace(-x_range, x_range, N[ii]).reshape(-1, 1)
        train_y = test_function(train_x).reshape(-1, 1)

        kriging.fit(train_x, train_y)
        test_x = np.linspace(train_x[0], train_x[-1], 10 * len(train_x)).reshape(-1, 1)
        test_y = test_function(test_x).reshape(-1, 1)

        y_predict = kriging.predict(test_x)
        error[ii] = np.mean((y_predict - test_y) ** 2)

        if ii == len(error) - 1:
            plt.figure()
            plt.scatter(train_x, train_y)
            plt.plot(test_x, y_predict, label='prediction')
            plt.plot(test_x, test_y, '--', label='truth')
            plt.legend()
            plt.show()
    plt.figure()
    plt.plot(N, error, '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel('Training points')
    plt.show()


def test_cokriging():
    hf_function = test_functions.circle
    x_range = (-1, 1.)

    cokrige = regression.CoKriging()
    krige = regression.Kriging()
    N = 2 ** np.arange(1, 3)
    error = np.zeros((len(N), 2))
    for ii in range(len(error)):
        x_lf = np.linspace(x_range[0], x_range[1], 11 + N[ii]).reshape(-1, 1)
        x_hf = np.linspace(x_range[0], x_range[1], N[ii]).reshape(-1, 1)
        y_lf = test_functions.forrester_low_fidelity(x_lf, y=hf_function(x_lf).reshape(-1, 1))
        y_hf = hf_function(x_hf).reshape(-1, 1)

        cokrige.fit(x_lf, y_lf, x_hf, y_hf)
        krige.fit(x_hf, y_hf)
        x_test = np.linspace(x_range[0], x_range[1], 20 * len(x_hf)).reshape(-1, 1)
        y_cokrige = cokrige.predict(x_test)
        y_krige = krige.predict(x_test)
        y_test = hf_function(x_test).reshape(-1, 1)
        y_test_lf = test_functions.forrester_low_fidelity(x_test, hf_function(x_test).reshape(-1, 1))

        error[ii, 0] = np.mean((y_krige - y_test) ** 2)
        error[ii, 1] = np.mean((y_cokrige - y_test) ** 2)
        if ii == len(error) - 1:
            plt.figure()
            plt.plot(x_test, y_krige, '-', label='Kriging')
            plt.plot(x_test, y_cokrige, '-', label='Co-Kriging')
            plt.plot(x_test, y_test, '--', label='High-Fidelity')
            plt.plot(x_test, y_test_lf, '--', label='Low-Fidelity')
            plt.plot(np.vstack((x_hf, x_lf)), np.vstack((y_hf, y_lf)), 'x', label='Training Data')
            plt.legend()
            plt.show()

    plt.figure()
    plt.plot(N, error, '--')
    plt.legend(('Kriging', 'Co-Kriging'))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel('Training points')
    plt.show()


def test_cokriging_nd():
    """ Test the co-Kriging class in N-dimensions """
    hf_function = test_functions.rastrigin
    x_range = (-1., 1.)
    n_dim = 2

    cokrige = regression.CoKriging()
    krige = regression.Kriging()
    n_points = 2 ** np.arange(3, 8)
    error = np.zeros((len(n_points), 2))
    for ii in range(len(error)):
        x_lf = np.random.uniform(x_range[0], x_range[1], size=(n_points[ii] * 10, n_dim))
        x_hf = np.random.uniform(x_range[0], x_range[1], size=(n_points[ii], n_dim))
        y_lf = test_functions.forrester_low_fidelity(x_lf, y=hf_function(x_lf)).reshape(-1, 1)
        y_hf = hf_function(x_hf).reshape(-1, 1)

        cokrige.fit(x_lf, y_lf, x_hf, y_hf)
        krige.fit(x_hf, y_hf)

        x_test = x_lf
        y_test = hf_function(x_test).reshape(-1, 1)
        y_cokrige = cokrige.predict(x_test)
        y_krige = krige.predict(x_test)

        error[ii, 0] = np.mean((y_krige - y_test) ** 2)
        error[ii, 1] = np.mean((y_cokrige - y_test) ** 2)
        if ii == len(error) - 1:
            plt.figure()
            plt.scatter(rad(x_lf), y_lf, label='Low-Fidelity')
            plt.scatter(rad(x_hf), y_hf, label='High-Fidelity')
            plt.scatter(rad(x_test), y_cokrige, label='Co-Kriging')
            plt.scatter(rad(x_test), y_krige, label='Kriging')
            plt.legend()
            plt.show()

    plt.figure()
    plt.plot(n_points, error, '--')
    plt.legend(('Kriging', 'Co-Kriging'))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel('Training points')
    plt.show()


def rad(x):
    return np.sqrt(np.sum(x ** 2, axis=1))


def test_deep_reinforcement():
    hf_function = test_functions.rastrigin
    x_range = (-5.12, 5.12)
    n_dim = 1

    model = regression.DeepReinforcement()

    n_points = 2 ** np.arange(6, 12)
    error = np.zeros((len(n_points), 2))
    for ii in range(len(error) - 1, len(error)):
        x_hf = np.random.uniform(x_range[0], x_range[1], size=(n_points[ii], n_dim))
        y_hf = hf_function(x_hf).reshape(-1, 1)

        model.fit(x_hf, y_hf)

        x_test = np.random.uniform(x_range[0], x_range[1], size=(n_points[ii] * 10, n_dim))
        x_test = np.sort(x_test, axis=0)
        y_test = hf_function(x_test).reshape(-1, 1)
        y_predict = model.predict(x_test)

        error[ii] = np.mean((y_predict - y_test) ** 2)
        if ii == len(error) - 1:
            plt.figure()
            plt.plot(x_test, y_predict, label='prediction')
            plt.plot(x_test, y_test, '--', label='truth')
            plt.legend()
            plt.show()

    plt.figure()
    plt.plot(n_points, error, '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel('Training points')
    plt.show()


if __name__ == "__main__":
    # test_rbf()
    # test_kriging()
    # test_cokriging()
    # test_cokriging_nd()
    test_deep_reinforcement()
