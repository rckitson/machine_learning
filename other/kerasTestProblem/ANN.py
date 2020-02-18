import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

X = np.random.uniform(-1, 1, size=(int(1e3), 2))
y = np.sum(X ** 2, axis=1)

for _ in range(5):
    lamReg = 0
    nodes = 2 ** np.random.randint(3, 5)
    layers = 2 ** np.random.randint(2, 3)
    learning_rate = 10 ** -3
    epochs = 50
    print('Nodes', nodes)
    print('Layers', layers)
    print('Learning Rate', learning_rate)
    print('Regularization', lamReg)

    act = 'relu'

    model = Sequential()
    model.add(Dense(units=nodes, activation=act, input_shape=(X.shape[1],), kernel_initializer='uniform'))
    for ii in range(layers - 1):
        model.add(Dense(units=nodes, activation=act, kernel_initializer='uniform'))
    model.add(Dense(units=1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=learning_rate))
    model.fit(X, y, validation_split=0.1, epochs=epochs,
              batch_size=10)
    y_predict = model.predict(X)

skip = max(1, len(X) // 20)
plt.figure()
plt.scatter(X[::skip, 0], y[::skip])
plt.scatter(X[::skip, 0], y_predict[::skip])
plt.show()
