import sys
import os 
import re
import numpy as np

path = os.path.expanduser('~')+'/closure/data'
inpath = path+'/tested/'
outpath = path+'/output/'
temp_path = path+'/temperature/'

if not os.path.exists(path+'/discard'):
    os.mkdir(path+'/discard')
if not os.path.exists(path+'/discard/temp/'):
    os.mkdir(path+'/discard/temp')
if not os.path.exists(path+'/discard/solid/'):
    os.mkdir(path+'/discard/solid')
if not os.path.exists(path+'/discard/two-phase/'):
    os.mkdir(path+'/discard/two-phase')

discard_temp = path+'/discard/temp/'
discard_solid = path+'/discard/solid/'
discard_twophase = path+'/discard/two-phase/'
# TODO: identify a method to discard glassy systems. Not critical unlikely to see classes in single particle systems
# will be signifcantly more important in investigations of two particle systems. possible approaches involve looking
# at the intermediate scattering function (glasses won't decay), estimating a non-gaussian parameter (caging effects),
# looking at MSD for evidence of caged diffusion.
# discard_glass = './discard/glass/'

files = os.listdir(outpath)
length = len(files)

# q=0 structure factor divergence (two-phase test)

for i in range(len(files)):
    test = re.findall('\d+', files[i])[0]
    output_fp = np.loadtxt(outpath+'output_'+test+'.dat')
    if np.max(output_fp[0,9]) > 1.0:
        # print('sample failed divergence test, assumed to be two phase.')
        os.rename(outpath+'output_'+test+'.dat', discard_twophase+'output_'+test+'.dat')

files = os.listdir(outpath)
print('{}/{} files failed the two-phase test'.strip().format((length-len(files)),length))
length = len(files)

# The structure factor doesn't satify the Hasen-Verlet rule.

for i in range(len(files)):
    test = re.findall('\d+', files[i])[0] 
    output_fp = np.loadtxt(outpath+'output_'+test+'.dat')
    if np.max(output_fp[:,9]) > 2.8:
        # print('sample failed hasen-verlet rule, assume to be solid.')
        os.rename(outpath+'output_'+test+'.dat', discard_solid+'output_'+test+'.dat')

files = os.listdir(outpath)
print('{}/{} files failed the Hasen-Verlet rule'.strip().format((length-len(files)),length))
length = len(files)

# Temperature doesn't fluctuate around specified value.

for i in range(len(files)):
    test = re.findall('\d+', files[i])[0] # [0] as it returns the numbers in a list. we can take the [0] element by naming convention
    temp_fp = np.loadtxt(temp_path+'temp_'+test+'.dat')
    temp = temp_fp[:,1]
    avg_temp = np.mean(temp)
    std_temp = np.sqrt(np.var(temp, axis=0, ddof=1)/len(temp))
    # if np.abs(avg_temp - 1.) > std_temp:
    if np.abs(avg_temp - 1.) > 0.01:
        # print('temperature not converged, non-equilibrium measurement')
        # print avg_temp, std_temp
        os.rename(outpath+'output_'+test+'.dat', discard_temp+'output_'+test+'.dat')

files = os.listdir(outpath)
print('{}/{} files failed the temperature test'.strip().format((length-len(files)),length))
length = len(files)




