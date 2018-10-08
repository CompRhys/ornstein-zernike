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
fig = plt.figure(figsize=(10,4.8))
ax = fig.add_subplot(111, projection='3d')
fig3, ax3 = plt.subplots()
# fig4, ax4 = plt.subplots()
# fig5, ax5 = plt.subplots()
# fig6, ax6 = plt.subplots()


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

for i in range(len(files)):
# for i in range(5):
    n = np.floor(int(re.findall('\d+', files[i])[0])/1000).astype(int)
    if np.any(used==n):
        output_fp = np.loadtxt(outpath+files[i])

        r   = output_fp[:,0]
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

        bridge_min = np.argmin(bridge)
        bridge = bridge[bridge_min:]
        trunc += bridge_min

        if max(bridge)>0.05:
            # os.rename(outpath+files[i], path+'/'+files[i])
            print files[i]

        # if min(bridge)<-1.4:
        #     # os.rename(outpath+files[i], path+'/'+files[i])
        #     print files[i]


        # plot points
        # if np.any(soft==n):
        #     ax.plot(g_r[trunc:] - 1., c_r[trunc:], bridge, linestyle="None", marker="o",
        #         color='darkorange', markersize=1.5)
        # if np.any(hard==n):
        #     ax.plot(g_r[trunc:] - 1., c_r[trunc:], bridge, linestyle="None", marker="o",
        #         color='teal', markersize=1.5)
        # if np.any(core==n):
        #     ax.plot(g_r[trunc:] - 1., c_r[trunc:], bridge, linestyle="None", marker="o",
        #         color='crimson', markersize=1.5)
        # if np.any(overlap==n):
        #     ax.plot(g_r[trunc:] - 1., c_r[trunc:], bridge, linestyle="None", marker="o",
        #         color='mediumblue', markersize=1.5)


        # gamma = g_r[trunc:] - 1. - c_r[trunc:]
        ax.plot(g_r[trunc:] - 1., c_r[trunc:], bridge,  marker="o", linestyle="None",
            color=test_colour[n-1], markersize=1.5)

        # ax3.plot(gamma, bridge, color=test_colour[n-1])#, linestyle="None", marker="x")
        # ax4.plot(c_r[trunc:], bridge, color=test_colour[n-1])#, linestyle="None", marker="x")
        # ax5.plot(phi[trunc:], bridge, color=test_colour[n-1])#, linestyle="None", marker="x")
        # ax6.plot(g_r[trunc:] - 1. , bridge, color=test_colour[n-1])#, linestyle="None", marker="x")



#configure axes
# patch = []
# for i in np.arange(len(potentials)):
#     if np.any(i+1==used):
#         patch.append(mpatches.Patch(color=test_colour[i], label=potentials[i]))

# hardsoft = []
# hardsoft.append(mpatches.Patch(color='b', label='Soft Core'))
# hardsoft.append(mpatches.Patch(color='orange', label='Hard Core'))

# ax.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))
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


ax3.set_xlabel('$\gamma(r)$')
ax3.set_ylabel('$B(r)$')
# ax4.set_xlabel('$c(r)$')
# ax4.set_ylabel('$B(r)$')
# ax5.set_xlabel('$\phi(r)$')
# ax5.set_ylabel('$B(r)$')
# ax6.set_xlabel('$h(r)$')
# ax6.set_ylabel('$B(r)$')

# ax.set_xlabel('$h(r)$')
# ax.set_ylabel('$c(r)$')
# ax.set_zlabel('$B(r)$')

ax.set_xlim([-1.5,1.5])
ax.set_ylim([-12,2])
ax.set_zlim([-1.1,0.1])


fig.tight_layout()
fig3.tight_layout()
# fig4.tight_layout()
# fig5.tight_layout()
# fig6.tight_layout()

plt.show()