import sys
import os 
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

def main():
    path = os.path.expanduser('~')+'/closure'
    outpath = path+'/data/passed/'
    files = os.listdir(outpath)
    fig, ax = plt.subplots(2,2, figsize=(16,9))


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

    for i in range(len(files)):
    # for i in range(5):
        regex = re.findall('\d+', files[i])
        if regex[-1] == '908':
            output_fp = np.loadtxt(outpath+files[i])
            print(regex)
            r   = output_fp[:,0]
            bridge = output_fp[:,1]
            h_r = output_fp[:,2]
            c_r = output_fp[:,3]
            
            ax[0,0].plot(r, h_r, label='{} {}'.format(regex[1], regex[-1]))
            ax[0,1].plot(r, c_r, label='{} {}'.format(regex[1], regex[-1]))
            ax[1,0].plot(r, bridge, label='{} {}'.format(regex[1], regex[-1]))

    # configure axes
    ax[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()