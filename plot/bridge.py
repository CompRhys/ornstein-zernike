import sys
import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

path = os.path.expanduser('~')+'/closure'
outpath = path+'/data/passed/'
inpath = path+'/data/tested/'
files = os.listdir(outpath)
fig = plt.figure(figsize=(10,4.8))
ax = fig.add_subplot(111, projection='3d')
# fig3, ax3 = plt.subplots()
# fig4, ax4 = plt.subplots()
# fig5, ax5 = plt.subplots()
# fig6, ax6 = plt.subplots()


test_colour = ['darkorange', 'lawngreen', 'teal', 
'darkseagreen', 'sienna', 'mediumorchid', 
'mediumturquoise', 'mediumvioletred', 'darkgoldenrod', 
'darkmagenta', 'red', 'forestgreen', 'mediumseagreen', 
'y', 'darkolivegreen', 'crimson', 'mediumpurple', 
'yellowgreen', 'mediumblue', 'coral', 'dimgrey']

potentials = ['lj', 'morse',                    # single minima
          'soft', 'yukawa', 'wca',          # repulsive
          'dlvo', 'exp-well',               # double minima
          'step', 'csw', 'rssaw',           # step potentials
          'gaussian', 'hat', 'hertzian',    # soft
          'llano']    

hard    = [1,2,5]
core    = [6,7,8,9,10]
overlap = [11,12,13]
soft    = [3,4]
tot  = [1,2,3,4,5,6,7,8,9,10,11,12,13]

used = tot

for i in range(len(files)):
# for i in range(5):
    n = np.floor(int(re.findall('\d+', files[i])[-1])/100).astype(int)
    if np.any(used==n):
        output_fp = np.loadtxt(outpath+files[i])

        r   = output_fp[:,0]
        bridge = output_fp[:,1]
        h_r = output_fp[:,2]
        c_r = output_fp[:,3]
        
        ax.plot(np.arcsinh(h_r), np.arcsinh(c_r), np.arcsinh(bridge),  marker="o", linestyle="None",
            color=test_colour[n-1], markersize=1.5)

        # ax3.plot(gamma, bridge, color=test_colour[n-1])#, linestyle="None", marker="x")
        # ax4.plot(c_r[trunc:], bridge, color=test_colour[n-1])#, linestyle="None", marker="x")
        # ax5.plot(phi[trunc:], bridge, color=test_colour[n-1])#, linestyle="None", marker="x")
        # ax6.plot(g_r[trunc:] - 1. , bridge, color=test_colour[n-1])#, linestyle="None", marker="x")



# configure axes
patch = []
for i in np.arange(len(potentials)):
    if np.any(i+1==used):
        patch.append(mpatches.Patch(color=test_colour[i], label=potentials[i]))

# hardsoft = []
# hardsoft.append(mpatches.Patch(color='b', label='Soft Core'))
# hardsoft.append(mpatches.Patch(color='orange', label='Hard Core'))

ax.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))
# ax2.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))
# # ax.legend(handles=hardsoft, loc='center left', bbox_to_anchor=(1, 0.5))
# # ax2.legend(handles=hardsoft, loc='center left', bbox_to_anchor=(1, 0.5))
# ax3.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))
# ax4.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))
# ax5.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))
# ax6.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))
# ax2.legend()
# ax3.legend()
# ax4.legend()


# ax3.set_xlabel('$\gamma(r)$')
# ax3.set_ylabel('$B(r)$')
# ax4.set_xlabel('$c(r)$')
# ax4.set_ylabel('$B(r)$')
# ax5.set_xlabel('$\phi(r)$')
# ax5.set_ylabel('$B(r)$')
# ax6.set_xlabel('$h(r)$')
# ax6.set_ylabel('$B(r)$')

# ax.set_xlabel('$h(r)$')
# ax.set_ylabel('$c(r)$')
# ax.set_zlabel('$B(r)$')

# ax.set_xlim([-1.5,1.5])
# ax.set_ylim([-12,2])
# ax.set_zlim([-1.1,0.1])


fig.tight_layout()
# fig3.tight_layout()
# fig4.tight_layout()
# fig5.tight_layout()
# fig6.tight_layout()

plt.show()