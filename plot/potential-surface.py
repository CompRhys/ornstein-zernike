import sys
import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as axes3d

path = os.path.expanduser('~')+'/Liquids'
outpath = path+'/data/output/'
inpath = path+'/data/tested/'
files = os.listdir(outpath)
fig = plt.figure(figsize=(10,4.8))
ax = fig.add_subplot(111, projection='3d')
fig3, ax3 = plt.subplots()


test_colour = ['darkorange', 'lawngreen', 'teal', 
'darkseagreen', 'sienna', 'mediumorchid', 
'mediumturquoise', 'mediumvioletred', 'darkgoldenrod', 
'darkmagenta', 'red', 'forestgreen', 'mediumseagreen', 
'y', 'darkolivegreen', 'crimson', 'mediumpurple', 
'yellowgreen', 'mediumblue', 'coral', 'dimgrey']

potentials = ['LJ', 'Soft Sphere', 'Morse', 'Pseudo-Hard', 
'Truncated Plasma', 'Yukawa', 'WCA', 'Step','CSW', 
'DLVO', 'DLVO-exp', 'Gaussian', 'Hat', 'Hertzian',
'RSSAW', 'Oscillating Decay']

used = [1,2,3,4,6,8,9,10,11,12,13,14,15,16]


for i in range(len(files)):
    output_fp = np.loadtxt(outpath+files[i])
    n = np.floor(int(re.findall('\d+', files[i])[0])/1000).astype(int)
    phi = output_fp[:,1]
    g_r = output_fp[:,2]
    # sigma_g_r = output_fp[:,3]
    c_r = output_fp[:,4]
    # sigma_c_r = output_fp[:,5]
    
    try:
        trunc_zero = np.max(np.where(g_r==0.0))+1
    except ValueError:
        trunc_zero = 0
    if n == 4:
        trunc_zero = np.min(np.where(g_r>1.0))+1
    trunc_small = 5
    trunc = max(trunc_small, trunc_zero)

    #plot points
    # if n == 16 : ax.plot(g_r[trunc:] - 1., c_r[trunc:], phi[trunc:], linestyle="None", marker="o",
    #         color=test_colour[n-1], markersize=1.5)

    ax.plot(g_r[trunc:] - 1., c_r[trunc:], phi[trunc:], linestyle="None", marker="o",
            color=test_colour[n-1], markersize=1.5)

# #configure axes
ax.set_xlim3d(-1,2)
ax.set_ylim3d(-10,3)
ax.set_zlim3d(-1,25)
# ax.legend()

X = np.linspace(-0.9999,2., 100)
Y = np.linspace(-10.,3.)
X, Y = np.meshgrid(X, Y)

# phi_hnc = ((X+1.)-np.log(X+1.)-Y-1.)
phi_py = np.log(np.abs(1.-Y/(X+1.)))

#plot points
# ax.plot_surface(X, Y, phi_hnc, color='b')
ax.plot_wireframe(X, Y, phi_py)

patch = []
for i in np.arange(len(potentials)):
    if np.any(i+1==used):
        patch.append(mpatches.Patch(color=test_colour[i], label=potentials[i]))
ax.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))


ax.set_xlabel('$h(r)$')
ax.set_ylabel('$c(r)$')
ax.set_zlabel('$phi(r)$')

fig.savefig(path+'/figures/phi.png')
fig.tight_layout()
plt.show()