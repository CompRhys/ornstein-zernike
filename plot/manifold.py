import sys
import os 
import re
import numpy as np
from mayavi import mlab

def main():
    # path = os.path.expanduser('~')+'/Liquids'
    path = os.path.expanduser('~')+'/masters/closure'

    outpath = path+'/data/old/output/'
    inpath = path+'/data/old/tested/'
    files = os.listdir(outpath)

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

            # sparse = np.max(np.where(r<10))+1
            sparse = np.argmin(r<7.0)
            l = 2
            d = 0.1
            p = 0.6


            if np.any(soft==n):
                if np.random.rand() >p: 
                    mlab.points3d((g_r[trunc:sparse])[::l] - 1., (c_r[trunc:sparse])[::l] , (bridge[:sparse-trunc])[::l], 
                    color=(0.9,0.4,0.1), line_width=1.0, scale_mode='none', scale_factor=d)
            if np.any(hard==n):
                if np.random.rand() > p: 
                    mlab.points3d((g_r[trunc:sparse])[::l] - 1., (c_r[trunc:sparse])[::l] , (bridge[:sparse-trunc])[::l], 
                    color=(0.7,0.1,0.4), line_width=1.0, scale_mode='none', scale_factor=d)
            if np.any(core==n):
                if np.random.rand() > p: 
                    mlab.points3d((g_r[trunc:sparse])[::l] - 1., (c_r[trunc:sparse])[::l] , (bridge[:sparse-trunc])[::l], 
                    color=(0.3,0.9,0.2), line_width=1.0, scale_mode='none', scale_factor=d)
            if np.any(overlap==n):
                if np.random.rand() > p: 
                    mlab.points3d((g_r[trunc:sparse])[::l] - 1., (c_r[trunc:sparse])[::l] , (bridge[:sparse-trunc])[::l], 
                    color=(0.5,0.6,0.9), line_width=1.0, scale_mode='none', scale_factor=d)

    # mlab.outline()
    mlab.show()

if __name__ == '__main__':
    main()