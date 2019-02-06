import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

def individual():
    path = os.path.expanduser('~') + '/closure'
    outpath = path + '/data/passed/'
    inpath = path + '/data/tested/'
    files = os.listdir(outpath)
    f = plt.figure(figsize=(10, 4.8))
    ax = f.add_subplot(111, projection='3d')
    fig, axes = plt.subplots(1, 2)


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
    tot = [1, 2, 3, 4, 5]#, 6, 7, 8, 9, 10, 11, 12, 13]

    used = tot

    for i in range(len(files)):
        # for i in range(5):
        n = np.floor(int(re.findall('\d+', files[i])[-1]) / 100).astype(int)
        if np.any(used == n):
            output_fp = np.loadtxt(outpath + files[i])

            r = output_fp[:, 0]
            bridge = output_fp[:, 1]
            h_r = output_fp[:, 3]
            c_r = output_fp[:, 4]

            ax.plot(np.arcsinh(h_r), np.arcsinh(c_r), np.arcsinh(bridge),
                    marker="o", #linestyle="None",
                    color=test_colour[n - 1], markersize=1.5)
            axes[0].plot(h_r, bridge, marker="o", linestyle="None",
                    color=test_colour[n - 1], markersize=.5)
            axes[1].plot(c_r, bridge, marker="o", linestyle="None",
                    color=test_colour[n - 1], markersize=.5)

    # configure axes
    patch = []
    for i in np.arange(len(potentials)):
        if np.any(i + 1 == used):
            patch.append(mpatches.Patch(color=test_colour[i], label=potentials[i]))


    g = np.array((0.031197, 0.50376, 1.83029, 1.92533, 1.245126,
              0.81728, 0.67208, 0.7227, 0.88436))
    h = g - 1
    c = np.array((-85.9399, -75.0357, -64.3664, -56.3433, -
                  50.56, -45.84, -41.84, -38.87, -35.67))
    bridge = np.array((-6.2421, -3.8702, -2.0794, -1.0561, -
                       0.5746, -0.3049, -0.2335, -0.0882, 0.0420))
    ax.plot(np.arcsinh(h), np.arcsinh(c), np.arcsinh(bridge), marker='o', markersize=3)

    ax.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()

    plt.show()

def group():
    path = os.path.expanduser('~') + '/closure'
    outpath = path + '/data/passed/'
    inpath = path + '/data/tested/'
    files = os.listdir(outpath)
    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(111, projection='3d')


    test_colour = ['C0', 'C1', 'C2', 'C3']

    potentials = ['hard','core-softened','soft','overlap']

    hard = [1, 2, 5]
    core = [6, 7, 8, 9, 10]
    overlap = [11, 12, 13]
    soft = [3, 4]
    tot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    used = tot

    for i in range(len(files)):
        # for i in range(5):
        n = np.floor(int(re.findall('\d+', files[i])[-1]) / 100).astype(int)
        output_fp = np.loadtxt(outpath + files[i])

        r = output_fp[:, 0]
        bridge = output_fp[:, 1]
        h_r = output_fp[:, 3]
        c_r = output_fp[:, 4]
        if np.any(hard == n):
            ax.plot(np.arcsinh(h_r), np.arcsinh(c_r), np.arcsinh(bridge),
                    marker="o", linestyle="None",
                    color=test_colour[0], markersize=1.5)
        elif np.any(core == n):
            ax.plot(np.arcsinh(h_r), np.arcsinh(c_r), np.arcsinh(bridge),
                    marker="o", linestyle="None",
                    color=test_colour[1], markersize=1.5)
        elif np.any(overlap == n):
            ax.plot(np.arcsinh(h_r), np.arcsinh(c_r), np.arcsinh(bridge),
                    marker="o", linestyle="None",
                    color=test_colour[3], markersize=1.5)
        elif np.any(soft == n):
            ax.plot(np.arcsinh(h_r), np.arcsinh(c_r), np.arcsinh(bridge),
                    marker="o", linestyle="None",
                    color=test_colour[2], markersize=1.5)

    # configure axes
    patch = []
    for i in np.arange(len(potentials)):
        patch.append(mpatches.Patch(color=test_colour[i], label=potentials[i]))


    ax.legend(handles=patch, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    individual()
    # group()
