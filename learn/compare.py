from __future__ import print_function
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import RBFSampler
import GPy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
# from mayavi import mlab
import numpy as np
import pickle
import time
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


path        = os.path.expanduser('~')+'/Liquids/src/learn'
test_path   = path+'/ml/test.dat'
test_set    = np.loadtxt(test_path)
test_size   = len(test_set)
# np.random.shuffle(test_set)
# test_size   = 1000

X           = test_set [:test_size,2:4]
X_nl        = test_set [:test_size,2:6]
phi         = test_set [:test_size,0]
bridge      = test_set [:test_size,1]
h           = test_set [:test_size,2]
c           = test_set [:test_size,3]
py 			= np.log(1. + h -c) + c - h

# Local Closures

# sgp_model_filename  = path+'/ml/sgp.pkl'
# sgp_model_pickle    = open(sgp_model_filename, 'rb')
# sgp                 = pickle.load(sgp_model_pickle)
# bridge_sgp, var_gp  = sgp.predict(X)#[0]
# bridge_sgp          = np.ravel(bridge_sgp)
# sgp_r2              = r2_score(bridge, bridge_sgp)
# sgp_model_pickle.close()

# print(sgp)

# print("Sparse GP score = {:.5f}".strip().format(sgp_r2))

# rbf_sample_filename = path+'/ml/rks-sample.pkl'
# rbf_sample_pickle   = open(rbf_sample_filename, 'rb')
# rbf_feature         = pickle.load(rbf_sample_pickle)
# rbf_sample_pickle.close()

# X_features          = rbf_feature.transform(X)

# sgd_model_filename  = path+'/ml/rks.pkl'
# sgd_model_pickle    = open(sgd_model_filename, 'rb')
# sgd                 = pickle.load(sgd_model_pickle)
# bridge_sgd          = sgd.predict(X_features)
# sgd_r2              = r2_score(bridge, bridge_sgd)
# sgd_model_pickle.close()

# print("SGDR score = {:.5f}".strip().format(sgd_r2))

mlp_model_filename  = path+'/ml/mlp.pkl'
mlp_model_pickle    = open(mlp_model_filename, 'rb')
mlp                 = pickle.load(mlp_model_pickle)
bridge_mlp          = mlp.predict(X)
mlp_r2              = r2_score(bridge, bridge_mlp)
mlp_model_pickle.close()

print("MLP score = {:.5f}".strip().format(mlp_r2))

# mlp_py_model_filename  = path+'/ml/mlp-py.pkl'
# mlp_py_model_pickle    = open(mlp_py_model_filename, 'rb')
# mlp_py                 = pickle.load(mlp_py_model_pickle)
# bridge_mlp_py          = np.ravel(mlp_py.predict(X)) + py
# mlp_py_r2              = r2_score(bridge, bridge_mlp_py)
# mlp_py_model_pickle.close()

# print("MLP - PY score = {:.5f}".strip().format(mlp_py_r2))

# Non-Local Closures

# sgp_nl_model_filename   = path+'/ml/nl-sgp.pkl'
# sgp_nl_model_pickle     = open(sgp_nl_model_filename, 'rb')
# sgp_nl                  = pickle.load(sgp_nl_model_pickle)
# bridge_sgp_nl           = sgp_nl.predict(X_nl)[0]
# bridge_sgp_nl           = np.ravel(bridge_sgp_nl)
# sgp_nl_r2               = r2_score(bridge, bridge_sgp_nl)
# sgp_nl_model_pickle.close()

# print("Non-Local Sparse GP score = {:.5f}".strip().format(sgp_nl_r2))

# nl_rbf_sample_filename  = path+'/ml/nl-rbf-sample.pkl'
# nl_rbf_sample_pickle    = open(nl_rbf_sample_filename, 'rb')
# nl_rbf_feature          = pickle.load(nl_rbf_sample_pickle)
# nl_rbf_sample_pickle.close()

# X_features_nl           = nl_rbf_feature.transform(X)

# sgd_nl_model_filename   = path+'/ml/nl-sgd.pkl'
# sgd_nl_model_pickle     = open(sgd_nl_model_filename, 'rb')
# sgd_nl                  = pickle.load(sgd_nl_model_pickle)
# bridge_sgd_nl           = sgd_nl.predict(X_features_nl)
# sgd_nl_r2               = r2_score(bridge, bridge_sgd_nl)
# sgd_nl_model_pickle.close()

# print("Non-Local SGDR score = {:.5f}".strip().format(sgd_nl_r2))

mlp_nl_model_filename   = path+'/ml/nl-mlp.pkl'
mlp_nl_model_pickle     = open(mlp_nl_model_filename, 'rb')
mlp_nl                  = pickle.load(mlp_nl_model_pickle)
bridge_mlp_nl           = mlp_nl.predict(X_nl)
mlp_nl_r2               = r2_score(bridge, bridge_mlp_nl)
mlp_nl_model_pickle.close()

print("Non-Local MLP score = {:.5f}".strip().format(mlp_nl_r2))

hnc_r2 = r2_score(bridge, np.zeros_like(bridge))
print("HNC score = {:.5f}".strip().format(hnc_r2))
mse_mlp=mse(bridge, bridge_mlp)
mse_mlp_nl=mse(bridge, bridge_mlp_nl)
mse_hnc=mse(bridge, np.zeros_like(bridge))

print(mse_hnc,mse_mlp, mse_mlp_nl)

# Pyplot 
plt.figure()
fig, ax = plt.subplots()
ax.plot(np.zeros_like(bridge), bridge,  linestyle="None", marker="x", markersize=3, label= 'HNC')
ax.plot(bridge_mlp, bridge,  linestyle="None", marker="x", markersize=3, label= 'Local MLP')
ax.plot(bridge_mlp_nl, bridge, linestyle="None", marker="x", markersize=3, label= 'Non-Local MLP')
# ax.plot(bridge_sgd, bridge, linestyle="None", marker="x", markersize=1.5, label= 'RKS')
# ax.plot(bridge_sgp, bridge, linestyle="None", marker="x", markersize=1.5, label= 'SGP')
# ax.plot(bridge_mlp_py, bridge,  linestyle="None", marker="x", markersize=3, label= 'MLP (PY reference)')
ax.plot([-2,2], [-2,2], linestyle="--", color='k')
ax.legend(markerscale=3, fontsize = 12)
ax.set_xlim([-1.05,0.1])
ax.set_ylim([-1.05,0.1])
ax.set_xlabel('$B_{ML}(r)$')
ax.set_ylabel('$B(r)$')
fig.tight_layout()


# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.plot(h, c, bridge,         linestyle="None", marker="o", markersize=1.5)

# # # Local

# # ax2.plot(h, c, bridge_sgp,     linestyle="None", marker="o", markersize=1.5)
# # ax2.plot(h, c, bridge_sgd,     linestyle="None", marker="o", markersize=1.5)
# # ax2.plot(h, c, bridge_mlp,     linestyle="None", marker="o", markersize=1.5)

# # # Non-Local

# # ax2.plot(h, c, bridge_sgp_nl,   linestyle="None", marker="o", markersize=1.5)
# # ax2.plot(h, c, bridge_sgd_nl,   linestyle="None", marker="o", markersize=1.5)
# ax2.plot(h, c, bridge_mlp_nl,   linestyle="None", marker="o", markersize=1.5)
# ax2.plot(h, c, ,   linestyle="None", marker="o", markersize=1.5)
# 
# fig2.tight_layout()

plt.show()

# MayaVi

# # mlab.points3d(h,c,bridge, mode='cube', color=(0.7,0.2,0.4), resolution=3, scale_mode='scalar', scale_factor=0.1)
# # mlab.points3d(h,c,bridge_mlp_nl-bridge, mode='cube', color=(0,0.5,0), resolution=3, scale_mode='scalar', scale_factor=0.1)
# # mlab.points3d(h,c,bridge_mlp-bridge, mode='cube', color=(0.0,0.0,0.0), resolution=3, scale_mode='scalar', scale_factor=0.1)
# # mlab.outline()
# # mlab.show()


