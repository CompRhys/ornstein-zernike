import sys
import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

path = os.path.expanduser('~')+'/Liquids'
outpath = path+'/data/output/'
inpath = path+'/data/tested/'
files = os.listdir(outpath)
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()


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

soft = [12,13,14]
used = [1,2,3,4,6,8,9,10,11,12,13,14,15]
# used = soft


for i in range(len(files)):
# for i in range(5):
    n = np.floor(int(re.findall('\d+', files[i])[0])/1000).astype(int)
    if np.any(used==n):

        output_fp = np.loadtxt(outpath+files[i])

        phi = output_fp[:,1]
        g_r = output_fp[:,2]
        c_r = output_fp[:,4]
        

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

        gamma = g_r[trunc:] - 1. - c_r[trunc:]
        h = g_r[trunc:] - 1.
        c = c_r[trunc:]

        phi_hnc = gamma - np.log(1+h)
        phi_py  = np.log(np.abs((1+gamma)/(1+h)))
        phi_msa = -c
        A = 5./6.
        B = 4./3.
        phi_ver = phi_hnc - 0.5 * A * gamma**2 / (1+0.5*B*gamma)

        ax2.plot(phi_hnc, phi[trunc:], color=test_colour[n-1])#, linestyle="None", marker="x")
        ax3.plot(phi_py, phi[trunc:], color=test_colour[n-1])#, linestyle="None", marker="x")
        ax4.plot(phi_msa, phi[trunc:], color=test_colour[n-1])#, linestyle="None", marker="x")

# #configure axes
patch = []
for i in np.arange(len(potentials)):
    if np.any(i+1==used):
        patch.append(mpatches.Patch(color=test_colour[i], label=potentials[i]))

# hardsoft = []
# hardsoft.append(mpatches.Patch(color='b', label='Soft Core'))
# hardsoft.append(mpatches.Patch(color='orange', label='Hard Core'))

# ax3.legend(handles=patch, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           # ncol=2, mode="expand", borderaxespad=0.)

ax2.plot([-2,13], [-2,13], linestyle="--", color='k')
ax3.plot([-2,13], [-2,13], linestyle="--", color='k')
ax4.plot([-2,13], [-2,13], linestyle="--", color='k')

ax3.set_xlabel('$\phi_{PY}(r)$')
ax2.set_xlabel('$\phi_{HNC}(r)$')
ax4.set_xlabel('$\phi_{MSA}(r)$')
ax4.set_ylabel('$\phi(r)$')
ax3.set_ylabel('$\phi(r)$')
ax2.set_ylabel('$\phi(r)$')

fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()

plt.show()