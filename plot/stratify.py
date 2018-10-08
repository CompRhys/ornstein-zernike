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


path    = os.path.expanduser('~')+'/Liquids'
outpath = path+'/data/output/'
inpath  = path+'/data/tested/'
mlpath  = path+'/src/learn'
files   = os.listdir(outpath)

fig, ax = plt.subplots(1,2, figsize=(12,4.8))

test_path   = mlpath+'/ml/test.dat'
test_set    = np.loadtxt(test_path)
test_size   = len(test_set)
# test_size   = 10

bridge      = test_set [:test_size,1]
X           = test_set [:test_size,2:4]
X_nl        = test_set [:test_size,2:6]
n           = test_set [:test_size,-1]

# mlp_model_filename = mlpath+'/ml/mlp.pkl'
mlp_model_filename  = mlpath+'/ml/nl-mlp.pkl'
mlp_model_pickle    = open(mlp_model_filename, 'rb')
mlp                 = pickle.load(mlp_model_pickle)
mlp_model_pickle.close()

b           = mlp.predict(X_nl)
# b           = mlp.predict(X)

test_colour = ['darkorange', 'lawngreen', 'teal', 
'darkseagreen', 'sienna', 'mediumorchid', 
'mediumturquoise', 'mediumvioletred', 'darkgoldenrod', 
'darkmagenta', 'red', 'forestgreen', 'mediumseagreen', 
'y', 'darkolivegreen', 'crimson', 'mediumpurple', 
'yellowgreen', 'mediumblue', 'coral', 'dimgrey']

hex_colour = ['#FF8C00','#7CFC00','#008080','#8FBC8F',
'#A0522D', '#BA55D3', '#48D1CC', '#C71585', '#B8860B',
'#8B008B', '#FF0000', '#228B22', '#3CB371', '#FFFF00',
'#556B2F', '#DC143C', '#9370DB', '#9ACD32', '#0000CD',
'#FF7F50', '#696969']

rgb_colour = [(255,140,0,255),(124,252,0,255),(0,128,128,255),(143,188,143,255),(160,82,45,255),(186,85,211,255),(72,209,204,255),
(199,21,133,255),(184,134,11,255),(139,0,139,255),(255,0,0,255),(34,139,34,255),(60,179,113,255),(192,190,4,255),
(85,107,47,255),(220,20,60,255),(147,112,219,255),(154,205,50,255),(0,0,205,255),(255,127,80,255),(105,105,105,255)]

rgba_colour = []

for j in np.arange(len(rgb_colour)):
    rgba_colour.append(np.array(rgb_colour[j])/255.)

potentials = ['Lennard Jones', 'Soft Sphere', 'Morse', 'Pseudo-Hard', 
'Truncated Plasma', 'Yukawa', 'WCA','Smooth-Step','CSW', 
'DLVO', 'DLVO-exp', 'Gaussian', 'Hat', 'Hertzian',
'RSSAW', 'Oscillating Decay']

hard    = [1,3,4]
core    = [8,9,10,11,15]
overlap = [12,13,14]
soft    = [2,6]
tot     = [1,2,3,4,6,8,9,10,11,12,13,14,15]

top     = overlap
bottom  = hard + core

mse_soft = 0
mse_hard = 0

b_soft = np.array([])
b_hard = np.array([])

bridge_soft = np.array([])
bridge_hard = np.array([])

colour_soft = []
colour_hard = []

for i in np.arange(len(b)):
    if np.any(top==n[i]):
        b_soft = np.append(b_soft, b[i])
        bridge_soft = np.append(bridge_soft, bridge[i])
        colour_soft.append(rgba_colour[int(n[i]-1)])

    if np.any(bottom==n[i]):
        b_hard = np.append(b_hard, b[i])
        bridge_hard = np.append(bridge_hard, bridge[i])
        colour_hard.append(rgba_colour[int(n[i]-1)])

mse_soft = ((b_soft-bridge_soft)**2).mean()
mse_hard = ((b_hard-bridge_hard)**2).mean()

print(mse_soft, mse_hard)

ax[0].scatter(b_soft, bridge_soft, color = colour_soft, s=3, marker="x")
ax[1].scatter(b_hard, bridge_hard, color = colour_hard, s=3, marker="x")

ax[0].plot([-2, 1], [-2, 1], linestyle="--", color='k')
ax[1].plot([-2, 1], [-2, 1], linestyle="--", color='k')
ax[0].set_xlim([-1.1,0.1])
ax[0].set_ylim([-1.1,0.1])
ax[1].set_xlim([-1.1,0.1])
ax[1].set_ylim([-1.1,0.1])

ax[0].set_xlabel('$B_{ML}(r)$')
ax[0].set_ylabel('$B_{Soft-Core}(r)$')
ax[1].set_xlabel('$B_{ML}(r)$')
ax[1].set_ylabel('$B_{Hard}(r)$')

fig.tight_layout()

plt.show()

