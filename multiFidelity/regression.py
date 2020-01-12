""" A module that defines kriging-based multi-fidelity methods """

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def gaussian(x, y, alpha=1.0):
    """ Gaussian kernel function
    Args:
        x: First point
        y: Second point
    Returns:
        Gaussian function value of the distance
    """
    return np.exp(-alpha * np.sum((x - y) ** 2))


def gaussian_correlation(X, alpha=1.0):
    """ Gaussian correlation between points in X

    Args:
        X: Input data
        alpha: Hyperparameter for the Gaussian radial basis function

    Returns:
        The correlation matrix
    """
    A = np.ones((len(X), len(X)))
    for ii in range(0, len(X)):
        for jj in range(ii, len(X)):
            A[ii, jj] = gaussian(X[ii, :], X[jj, :], alpha)
            A[jj, ii] = A[ii, jj]
    return A


class Regression:
    """ A general class for the regression models """

    def __init__(self, scale=True):
        """ Constructor

        Args:
            scale: Scale the data
        """

        if scale:
            self.scale_x = MinMaxScaler(feature_range=(-0.5, 0.5))
            self.scale_y = MinMaxScaler(feature_range=(-0.5, 0.5))
        else:
            self.scale_x = None


class RadialBasisFunction(Regression):
    """ Radial Basis Function model """

    def __init__(self, scale=True):
        super().__init__(scale)
        print('RBF Initialized')

    def fit(self, X, y):
        """ Fit the model to the data

        Args:
            X: Input data
            y: Output data
        """
        X = self.scale_x.fit_transform(X)
        y = self.scale_y.fit_transform(y)
        self.x = X
        self.y = y

        residual = np.zeros(20)
        alpha_search = 10 ** np.linspace(-3, 3, len(residual))
        print('Optimizing alpha')
        for ii, alpha in enumerate(alpha_search):
            correlation = gaussian_correlation(X, alpha=alpha)
            correlation_fit = np.vstack((correlation, np.ones((1, len(correlation)))))
            y_fit = np.vstack((y, np.ones((1, 1))))
            weights = np.linalg.lstsq(correlation_fit, y_fit, rcond=None)[0]
            residual[ii] = np.linalg.norm(np.matmul(correlation_fit, weights) - y_fit)
        self.alpha = alpha_search[np.argmin(residual)]
        print(self.alpha)

        correlation = gaussian_correlation(X, alpha=self.alpha)

        correlation_fit = np.vstack((correlation, np.ones((1, len(correlation)))))
        y_fit = np.vstack((y, np.ones((1, 1))))
        self.weights = np.linalg.lstsq(correlation_fit, y_fit, rcond=None)[0]

    def predict(self, X):
        """ Predict the output value given a new input

        Args:
            X: Input data

        Returns:
            The predictions at X
        """
        X = self.scale_x.transform(X)

        distances = np.zeros((len(X), len(self.x)))
        for ii in range(len(X)):
            for jj in range(len(self.x)):
                distances[ii, jj] = gaussian(X[ii], self.x[jj], self.alpha)

        y_predict = np.matmul(distances, self.weights)
        return self.scale_y.inverse_transform(y_predict)


class Kriging(Regression):
    """ An ordinary Kriging class

        This class fits a Gaussian process to residual from
        a least-squares linear model
    """

    def __init__(self, scale=True):
        """ Constructor

        Args:
            scale: Whether or not to scale the data
        """
        super().__init__(scale)
        self.beta = None
        self.radial_basis_function = RadialBasisFunction()
        print('Kriging initialized')

    def fit(self, X, y):
        """ Fit a Kriging model

        Args:
            X: The independent data
            y: The dependent data
        """
        if self.scale_x:
            X = self.scale_x.fit_transform(X)
            y = self.scale_y.fit_transform(y)

        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = y - np.matmul(X, self.beta)

        self.radial_basis_function.fit(X, y_hat)

    def predict(self, X):
        """ Predict the function value given X

        Args:
            X: The new data

        Returns:
            The predicted function value at X
        """
        if self.scale_x:
            X = self.scale_x.transform(X)

        y_hat = self.radial_basis_function.predict(X)
        y_predict = y_hat + np.matmul(X, self.beta)
        if self.scale_x:
            return self.scale_y.inverse_transform(y_predict)
        return y_predict


class CoKriging(Regression):
    """ A co-Kriging class for multi-fidelity modeling """

    def __init__(self, scale=True):
        """ Constructor """
        super().__init__(scale)
        self._low_fidelity_rbf = Kriging()
        self._delta_rbf = Kriging()
        print('Co-Kriging initialized')

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        """ Fit the model to the low fidelity and high fidelity data

        Args:
            X_lf: The low-fidelity independent data
            y_lf: The low-fidelity dependent data
            X_hf: The high-fidelity independent data
            y_hf: The high-fidelity dependent data
        """
        if self.scale_x:
            X_lf = self.scale_x.fit_transform(X_lf)
            X_hf = self.scale_x.transform(X_hf)
            y_lf = self.scale_y.fit_transform(y_lf)
            y_hf = self.scale_y.transform(y_hf)

        self._low_fidelity_rbf.fit(X_lf, y_lf)
        y_lf_predict = self._low_fidelity_rbf.predict(X_hf)
        self.rho = np.linalg.lstsq(y_lf_predict, y_hf, rcond=0)[0]
        self._delta_rbf.fit(X_hf, y_hf - self.rho * y_lf_predict)

    def predict(self, X):
        """ Make a prediction based on the low fidelity and high fidelity data

        Args:
            X: The query points
        Returns:
            The predicted values
        """
        if self.scale_x:
            X = self.scale_x.transform(X)

        y_lf_predict = self._low_fidelity_rbf.predict(X)
        delta = self._delta_rbf.predict(X)
        y_predict = delta + self.rho * y_lf_predict

        return self.scale_y.inverse_transform(y_predict)


class DeepReinforcement(Regression):
    """ A class for using a deep neural network to approximate the function """

    def __init__(self, scale=True):
        super().__init__(scale)
        self.model = None

    def fit(self, x, y):
        """ Fit the model to the data

        Args:
            x: Input data
            y: Output data
        """
        if self.scale_x:
            x = self.scale_x.fit_transform(x)
            y = self.scale_y.fit_transform(y)

        param = []
        residual = np.zeros(10)
        for ii in range(len(residual)):
            nodes = 2 ** np.random.randint(4, 5)
            layers = max(nodes, 2 ** np.random.randint(4, 5))
            learning_rate = 10 ** np.random.uniform(-4, -2)
            param.append([nodes, layers, learning_rate])
            print('Nodes, layers, learning rate:', param[-1])

            history = self.build_model(x, y, nodes, layers, learning_rate=learning_rate)
            residual = np.min(history.history['loss'])
        best_param = param[np.argmin(residual)]
        print('Best param', best_param)
        self.build_model(x, y, best_param[0], best_param[1], learning_rate=best_param[2], epochs=100)

    def build_model(self, x, y, nodes, layers, epochs=10, learning_rate=1e-3):
        activation_function = 'relu'

        self.model = Sequential()
        self.model.add(
            Dense(units=nodes, activation=activation_function, input_shape=(x.shape[1],),
                  kernel_initializer='lecun_normal'))
        for ii in range(layers - 1):
            self.model.add(Dense(units=nodes, activation=activation_function, kernel_initializer='lecun_normal'))
        self.model.add(Dense(units=1, kernel_initializer='lecun_normal'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

        history = self.model.fit(x, y, validation_split=0.1, epochs=epochs,
                                 batch_size=3)
        return history

    def predict(self, x):
        """ Make a prediction

        Args:
            x: Input data

        Returns:
            The predicted value

        """
        x = self.scale_x.transform(x)
        y_predict = self.model.predict(x)

        if self.scale_x:
            return self.scale_y.inverse_transform(y_predict)
        return y_predict
