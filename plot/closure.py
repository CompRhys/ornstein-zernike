import sys
import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as axes3d

import pickle
from sknn.mlp import Regressor, Layer
import GPy
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


path = os.path.expanduser('~')+'/Liquids'
outpath = path+'/data/output/'
inpath = path+'/data/tested/'
mlpath = path+'/src/learn'
files = os.listdir(outpath)
# fig = plt.figure(figsize=(10,4.8))
# ax = fig.add_subplot(111, projection='3d')
fig2, ax2 = plt.subplots(2, figsize=(6,10))
# fig3, ax3 = plt.subplots()

mlp_model_filename = mlpath+'/ml/mlp.pkl'
# mlp_model_filename = mlpath+'/ml/nl-mlp.pkl'
# mlp_model_filename = mlpath+'/ml/nl-sgp.pkl'
# mlp_model_filename = mlpath+'/ml/sgp.pkl'
mlp_model_pickle = open(mlp_model_filename, 'rb')
mlp         = pickle.load(mlp_model_pickle)
mlp_model_pickle.close()

# rbf_sample_filename = mlpath+'/ml/rbf-sample.pkl'
# rbf_sample_pickle   = open(rbf_sample_filename, 'rb')
# rbf_feature         = pickle.load(rbf_sample_pickle)
# rbf_sample_pickle.close()

# sgd_model_filename  = mlpath+'/ml/rks.pkl'
# sgd_model_pickle    = open(sgd_model_filename, 'rb')
# sgd                 = pickle.load(sgd_model_pickle)

test_colour = ['darkorange', 'lawngreen', 'teal', 
'darkseagreen', 'sienna', 'mediumorchid', 
'mediumturquoise', 'mediumvioletred', 'darkgoldenrod', 
'darkmagenta', 'red', 'forestgreen', 'mediumseagreen', 
'y', 'darkolivegreen', 'crimson', 'mediumpurple', 
'yellowgreen', 'mediumblue', 'coral', 'dimgrey']

potentials = ['Lennard Jones', 'Soft Sphere', 'Morse', 'Pseudo-Hard', 
'Truncated Plasma', 'Yukawa', 'WCA','Smooth-Step','CSW', 
'DLVO', 'DLVO-exp', 'Gaussian', 'Hat', 'Hertzian',
'RSSAW', 'Oscillating Decay']

hard    = [1,3,4]
core    = [8,9,10,11,15]
overlap = [12,13,14]
soft    = [2,6]
tot  = [1,2,3,4,6,8,9,10,11,12,13,14,15]

used = tot
top  = overlap
bottom = hard + core

mse_soft = 0
mse_hard = 0
num_soft = 0
num_hard = 0

for i in range(len(files)):
# for i in range(5):
    n = np.floor(int(re.findall('\d+', files[i])[0])/1000).astype(int)
    if np.any(used==n):

        output_fp = np.loadtxt(outpath+files[i])

        r   = output_fp[:,0]
        phi = output_fp[:,1]
        g_r = output_fp[:,2]
        c_r = output_fp[:,4]
        dg  = np.gradient(g_r, r[0])
        dc  = np.gradient(c_r, r[0])

        try:
            trunc_zero = np.max(np.where(g_r==0.0))+1
        except ValueError:
            trunc_zero = 0
        if n == 4:
            trunc_zero = np.min(np.where(g_r>1.0))+1
        trunc_small = 5
        trunc = max(trunc_small, trunc_zero)


        # the difference between the two methods is within floating point error.
        bridge = np.log(g_r[trunc:])+c_r[trunc:]+1.-g_r[trunc:]+phi[trunc:]
        # bridge = np.log(g_r[trunc:]*np.exp(phi[trunc:]))+c_r[trunc:]+1.-g_r[trunc:]

        bridge_min = np.argmin(bridge)+2
        bridge = bridge[bridge_min:]
        trunc += bridge_min

        sparse = np.max(np.where(r<10))+1

        h = g_r[trunc:sparse] - 1.
        c = c_r[trunc:sparse]
        dh = dg[trunc:sparse]
        dc = dc[trunc:sparse]
        bridge = bridge[:sparse-trunc]


        test_set    = np.transpose(np.array((h,c)))
        # test_set    = np.transpose(np.array((h,c,dh,dc)))
        # b  = np.ravel(mlp.predict(test_set)[0])
        b  = np.ravel(mlp.predict(test_set))

        # X_features = rbf_feature.transform(test_set)
        # b  = np.ravel(sgd.predict(X_features))

        # ax.plot(h, c, phi[trunc:],  marker="o", linestyle="None",
        #     color=test_colour[n-1], markersize=1.5)
        # ax3.plot(phi_ver, phi[trunc:], color=test_colour[n-1])#, linestyle="None", marker="x")
        # ax2.plot(b, bridge, color=test_colour[n-1], linestyle="None", marker="x", markersize=3)
        # ax2.plot([bridge.min(), bridge.max()], [bridge.min(), bridge.max()], linestyle="--", color='k')
        # ax3.pl

        

        if np.any(top==n):
            ax2[0].plot(b, bridge, color=test_colour[n-1], linestyle="None", marker="x", markersize=3)
            mse_soft += ((b-bridge)**2).sum().copy()
            num_soft += len(b)

        if np.any(bottom==n):
            ax2[1].plot(b, bridge, color=test_colour[n-1], linestyle="None", marker="x", markersize=3)
            mse_hard += ((b-bridge)**2).sum()
            num_hard += len(b)

print('soft', mse_soft/num_soft, num_soft)

print('hard', mse_hard/num_hard, num_hard)

ax2[0].plot([-2, 1], [-2, 1], linestyle="--", color='k')
ax2[1].plot([-2, 1], [-2, 1], linestyle="--", color='k')
ax2[0].set_xlim([-1.05,0.1])
ax2[0].set_ylim([-1.05,0.1])
ax2[1].set_xlim([-1.05,0.1])
ax2[1].set_ylim([-1.05,0.1])


# #configure axes
# patch = []
# for i in np.arange(len(potentials)):
#     if np.any(i+1==used):
#         patch.append(mpatches.Patch(color=test_colour[i], label=potentials[i]))

# # hardsoft = []
# # hardsoft.append(mpatches.Patch(color='b', label='Soft Core'))
# # hardsoft.append(mpatches.Patch(color='orange', label='Hard Core'))

# # ax.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))
# ax2.legend(handles=patch, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=4, mode="expand", borderaxespad=0.)

# ax3.set_xlabel('$\phi_{PY}(r)$')
# ax3.set_ylabel('$\phi(r)$')
ax2[0].set_xlabel('$B_{ML}(r)$')
ax2[0].set_ylabel('$B_{Soft-Core}(r)$')
ax2[1].set_xlabel('$B_{ML}(r)$')
ax2[1].set_ylabel('$B_{Hard}(r)$')


# ax.set_xlabel('$h(r)$')
# ax.set_ylabel('$c(r)$')
# ax.set_zlabel('$B(r)$')

# fig.tight_layout()
fig2.tight_layout()
# fig3.tight_layout()

plt.show()