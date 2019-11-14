import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

from mpl_toolkits.mplot3d import Axes3D

data_path = sys.argv[1]

data_set = np.loadtxt(data_path, delimiter=",", skiprows=1)
X = data_set[:, 2:4]
y = data_set[:, 1]
# n = data_set[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(12, input_dim=2, kernel_initializer='normal', activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

print("Learning MLP model with {} training points".format(len(y_train)))

# define early stopping callback
# earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10, \
#                           verbose=1, mode='auto')
# callbacks_list = [earlystop]
callbacks_list = []

try:
    history = model.fit(X_train, y_train, epochs=300, batch_size=64, \
                        callbacks=callbacks_list, verbose=1, \
                        validation_split=0.2)
except KeyboardInterrupt:
    pass

model.save('./local.h5')  # creates a HDF5 file 'my_model.h5'


print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

y_predicted = model.predict(X_test)

r2          = r2_score(y_test, y_predicted)
r2_hnc          = r2_score(y_test, np.zeros_like(y_predicted))
print(r2, r2_hnc)

num = 50
x = np.linspace(-1, 2, num)
y = np.linspace(-5, 2, num)
xx, yy = np.meshgrid(x,y)
points = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
y_surf = model.predict(points)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(points[:,0], points[:,1], y_surf.reshape(-1,))
ax.plot(X_train[:,0], X_train[:,1], y_train.reshape(-1,), marker="o", markersize=0.5, linestyle="None",)


plt.show()