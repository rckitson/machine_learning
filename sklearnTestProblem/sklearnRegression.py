import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split

scaler = MinMaxScaler()
dat = np.loadtxt('spring.csv', delimiter=',')

X = scaler.fit_transform(dat[:, 0].reshape(-1,1))
y = scaler.fit_transform(dat[:, 1].reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Ridge
from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print 'Ridge', np.mean(scores)

plt.figure()
plt.plot(X, y)
plt.scatter(X_test, y_predict, label=('Ridge %.2f' % (-1*np.mean(scores))))

# K neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors = len(X)/10)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print 'KNN', np.mean(scores)

plt.scatter(X_test, y_predict, label=('KNN %.2f' % (-1*np.mean(scores))))
plt.legend()

# Decision Treen
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print 'Tree', np.mean(scores)

plt.scatter(X_test, y_predict, label=('Tree %.2f' % (-1*np.mean(scores))))
plt.legend()

# Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
model = GaussianProcessRegressor(kernel=RBF(0.1))
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print 'GP', np.mean(scores)

plt.scatter(X_test, y_predict, label=('GP %.2f' % (-1*np.mean(scores))))
plt.legend()

# Neural Network
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100,100,), activation='relu')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores = cross_val_score(model, X, np.ravel(y), cv=10, scoring='neg_mean_squared_error')
print 'MLP', np.mean(scores)

plt.scatter(X_test, y_predict, label=('MLP %.2f' % (-1*np.mean(scores))))
plt.legend()

# Support Vector Regression
from sklearn import svm
model = svm.SVR(tol=1e-6, degree=len(X)/10)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print 'SVR', np.mean(scores)

plt.scatter(X_test, y_predict, label=('SVR %.2f' % (-1*np.mean(scores))))
plt.legend()
plt.savefig('sklearnRegression.pdf')


