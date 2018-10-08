import os
import numpy as np
import pickle
import time
from sknn.mlp import Regressor, Layer
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse


path = os.path.expanduser('~')+'/Liquids/src/learn'
train_path = path+'/ml/train.dat'

training_set    = np.loadtxt(train_path)
train_size      = len(training_set)
# # shuffle because sometimes we can use whole set due to memory
# np.random.shuffle(training_set) 
# train_size      = 5000

X = training_set[:train_size, 2:4]
Y = training_set[:train_size, 1]
n = training_set[:train_size, -1]

hard    = [1,3,4]
core    = [8,9,10,11,15]
overlap = [12,13,14]
soft    = [2,6]
tot     = [1,2,3,4,6,8,9,10,11,12,13,14,15]

# used    = core

# for i in np.arange(train_size):
#     if np.any(used==n[i]):
#         if 'X_res' in dir() :
#             # print(X_res)
#             X_res = np.vstack((X_res, X[i,:]))
#             # print(X_res)
#             # print(Y_res)
#             Y_res = np.hstack((Y_res, Y[i]))
#             # print(Y_res)

#         else:
#             X_res = X[i,:]
#             Y_res = Y[i]

# X = X_res
# Y = Y_res

print("Learning MLP model with %i training points" %train_size)

nn = Regressor(
    layers=[
        # Layer("ExpLin", units=128),
        # Layer("Tanh", units=16),
        Layer("Rectifier", units=16),
        Layer("Rectifier", units=16),
        Layer("Rectifier", units=16),
        Layer("Rectifier", units=16),
        Layer("Rectifier", units=16),
        Layer("Linear")],
    random_state=8,
    learning_rule='adam',
    learning_rate = 0.001,
    # learning_rule=u'sgd', 
    # learning_rate=0.01, 
    # learning_momentum=0.9,
    batch_size=200,
    n_iter=2000,
    valid_size=0.1,
    regularize='L2',
    weight_decay=5e-5,
    verbose=True)

# gs = GridSearchCV(nn, param_grid={
#     'learning_rate': [0.001],
#     'hidden0__units': [4, 8, 12],
#     'hidden1__units': [4, 8, 12],
#     'hidden2__units': [4, 8, 12],
#     })

t0 = time.time()

try:
    # gs.fit(X, Y)
    nn.fit(X, Y)
except KeyboardInterrupt:
    pass
mlp_fit = time.time() - t0
print("MLP model fitted in {:.3f} s".strip().format(mlp_fit))

nn_r2       = nn.score(X, Y)
print(nn_r2)

mlp_model_filename = path+'/ml/mlp.pkl'
mlp_model_pickle = open(mlp_model_filename, 'wb')
# pickle.dump(gs, mlp_model_pickle)
pickle.dump(nn, mlp_model_pickle)
mlp_model_pickle.close()


test_path   = path+'/ml/test.dat'
test_set    = np.loadtxt(test_path)
test_size   = len(test_set)
# test_size   = 10000

X_test      = test_set [:test_size,2:4]
bridge 		= test_set [:test_size,1]
h           = test_set [:test_size,2]
c           = test_set [:test_size,3]

bridge_mlp  = np.ravel(nn.predict(X_test))
nn_r2       = nn.score(X_test, bridge)
mse_mlp     =mse(bridge, bridge_mlp)

print(nn_r2, mse_mlp)
# bridge_mlp  = np.ravel(gs.predict(X_test))
# print(gs.score(X_test, bridge))
# print( gs.best_params_)


fig1, ax1   = plt.subplots()
ax1.plot(bridge_mlp, bridge, linestyle="None", marker="o", markersize=.8)
ax1.plot([bridge.min(), bridge.max()], [bridge.min(), bridge.max()], linestyle="--", color='k')
fig1.tight_layout()

plt.show()