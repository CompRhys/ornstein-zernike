import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

from mpl_toolkits.mplot3d import Axes3D

train_path = sys.argv[1]
test_path = sys.argv[2]

training_set    = np.loadtxt(train_path)
train_size      = len(training_set)

X_train = training_set[:train_size, 2:4]
y_train = training_set[:train_size, 0]

model = Sequential()
model.add(Dense(12, input_dim=2, kernel_initializer='normal', activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='logcosh', optimizer='sgd')

print("Learning MLP model with %i training points" %train_size)

# define early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10, \
                          verbose=1, mode='auto')
callbacks_list = [earlystop]

try:
    history = model.fit(X_train, y_train, epochs=100, batch_size=128, \
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


test_set    = np.loadtxt(test_path)
test_size   = len(test_set)
# test_size   = 10000

X_test      = test_set[:test_size,2:4]
bridge      = test_set[:test_size,0]

Y_predicted = model.predict(X_test)

r2          = r2_score(bridge, Y_predicted)
print(r2)

num = 50
x = np.linspace(-1, 2, num)
y = np.linspace(-30, 2, num)
xx, yy = np.meshgrid(x,y)
points = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
y_surf = model.predict(points)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(points[:,0], points[:,1], y_surf.reshape(-1,))
ax.plot(X_train[:,0], X_train[:,1], y_train.reshape(-1,), marker="o", markersize=0.5, linestyle="None",)


plt.show()