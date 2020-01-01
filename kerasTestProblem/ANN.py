import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dat = np.loadtxt('spring.csv', delimiter=',')
# Scale to [0,1]

X = np.linspace(dat[0,0], dat[-1,0], 1e4)
y = np.interp(X, dat[:,0], dat[:,1])
shuff = np.random.permutation(len(X))
X = (X[shuff] - min(X))/np.ptp(X)
y = (y[shuff] - min(y))/np.ptp(y)

# X = np.linspace(0,1.0,10000).reshape(-1,1)
# y = np.sin(2*np.pi*3/2.*X)*(1/2.*(1 - np.cos(2*np.pi*X))) + np.random.rand(len(X)).reshape(-1,1)/10.0
# print X, y
# shuff = np.random.permutation(len(X))
# X = (X[shuff] - min(X))/np.ptp(X)
# y = (y[shuff] - min(y))/np.ptp(y)

print len(X), 'samples'

lamReg = 0.0
nodes = 2**5
layers = 2**1
print 'Nodes', nodes
print 'Layers', layers

act = 'elu'

model = Sequential()
model.add(Dense(units=nodes, activation=act, input_shape=(1,), kernel_initializer='uniform'))
for ii in range(layers-1):
    model.add(Dense(units=nodes, activation=act, kernel_initializer='uniform'))
model.add(Dense(units=1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam())
model.fit(X, y, validation_split=0.1, shuffle=True, callbacks=[EarlyStopping(patience=10)], epochs=300, batch_size=100)
y_predict = model.predict(X)

plt.figure()
plt.scatter(X[::len(X)/200], y[::len(X)/200])
plt.scatter(X[::len(X)/200], y_predict[::len(X)/200])
plt.show()

