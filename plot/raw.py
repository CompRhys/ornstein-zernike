import os
import numpy as np
import matplotlib.pyplot as plt

# Notation

# read data 
path = os.path.expanduser('~')+'/masters/closure/data/raw/'
test_number = str(input("Input file number: "))
rdf = np.loadtxt(path+'rdf_'+test_number+'.dat')
sq = np.loadtxt(path+'sq_'+test_number+'.dat')
phi = np.loadtxt(path+'phi_'+test_number+'.dat')
temp = np.loadtxt(path+'temp_'+test_number+'.dat')

print(rdf.shape)

# # Plot

fig, axes = plt.subplots(2,2)
axes[0,0].plot(rdf[:,0], rdf[:,1:], 'b--', linewidth=0.5)
axes[0,0].plot(rdf[:,0], np.mean(rdf[:,1:], axis=1), 'r', linewidth=2)
axes[0,0].set_xlabel('r/$\sigma$')
axes[0,0].set_ylabel('$g(r)$')
# axes[0,1].set_xlim([0,2])
axes[0,1].plot(sq[:,0], sq[:,1:], 'b--', linewidth=0.5)
axes[0,1].plot(sq[:,0], np.mean(sq[:,1:], axis=1), 'r', linewidth=2)
axes[0,1].set_xlabel('r/$\sigma$')
axes[0,1].set_ylabel('$c(r)$') 
axes[1,0].plot(phi[:,0], phi[:,1], 'b--', linewidth=0.5)
axes[1,0].set_xlabel('r/$\sigma$')
axes[1,0].set_ylabel('$c(r)$')
axes[1,0].set_ylim([-3,6])
axes[1,1].plot(temp[:,0], temp[:,1], 'b--', linewidth=0.5)
axes[1,1].set_xlabel('r/$\sigma$')
axes[1,1].set_ylabel('$c(r)$')

# plt.savefig(path+'/figures/gr'+test_number+'.png')

plt.show()
