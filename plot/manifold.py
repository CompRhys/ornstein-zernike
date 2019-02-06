import sys
import os
import re
import numpy as np
from mayavi import mlab


def main():

    path = os.path.expanduser('~') + '/closure'
    outpath = path + '/data/passed/'
    inpath = path + '/data/tested/'

    files = os.listdir(outpath)

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

    hard = [1, 2, 5]
    core = [6, 7, 8, 9, 10]
    overlap = [11, 12, 13]
    soft = [3, 4]
    tot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    used = tot

    for i in range(len(files)):
        # for i in range(5):
        n = np.floor(int(re.findall('\d+', files[i])[-1]) / 100).astype(int)
        if np.any(used == n):
            output_fp = np.loadtxt(outpath + files[i])

            r = output_fp[:, 0]
            bridge = np.arcsinh(output_fp[:, 1])
            h_r = np.arcsinh(output_fp[:, 2])
            c_r = np.arcsinh(output_fp[:, 3])

            sparse = np.argmin(r < 5.0)
            l = 2
            d = 0.1
            p = 0.6

            if np.any(soft == n):
                if np.random.rand() > p:
                    mlab.points3d((h_r[:sparse])[::l], (c_r[:sparse])[::l],
                                  (bridge[:sparse])[::l], color=(0.9, 0.4, 0.1),
                                  line_width=1.0, scale_mode='none', scale_factor=d)
            if np.any(hard == n):
                if np.random.rand() > p:
                    mlab.points3d((h_r[:sparse])[::l], (c_r[:sparse])[::l],
                                  (bridge[:sparse])[::l], color=(0.7, 0.1, 0.4),
                                  line_width=1.0, scale_mode='none', scale_factor=d)
            if np.any(core == n):
                if np.random.rand() > p:
                    mlab.points3d((h_r[:sparse])[::l], (c_r[:sparse])[::l],
                                  (bridge[:sparse])[::l], color=(0.3, 0.9, 0.2),
                                  line_width=1.0, scale_mode='none', scale_factor=d)
            if np.any(overlap == n):
                if np.random.rand() > p:
                    mlab.points3d((h_r[:sparse])[::l], (c_r[:sparse])[::l],
                                  (bridge[:sparse])[::l], color=(0.5, 0.6, 0.9),
                                  line_width=1.0, scale_mode='none', scale_factor=d)

    # mlab.outline()
    mlab.show()

if __name__ == '__main__':
    main()
