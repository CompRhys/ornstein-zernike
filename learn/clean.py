import sys
import os 
import re
import numpy as np

def clean(directory, passed, failed):
    """
    TODO: identify a method to discard glassy systems. Not critical unlikely 
    to see classes in single particle systems will be signifcantly more 
    important in investigations of two particle systems. possible approaches 
    involve looking at the intermediate scattering function (glasses won't 
    decay), estimating a non-gaussian parameter (caging effects), looking at 
    MSD for evidence of caged diffusion.
    """

    if not os.path.exists(passed):
        os.mkdir(passed)
    if not os.path.exists(failed):
        os.mkdir(failed)

    files = os.listdir(directory)
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            test_case = np.loadtxt(os.path.abspath(os.path.join(dirpath, file)))
            # q=0 structure factor divergence (two-phase test)
            if test_case[0,-1] > 1.0:
                os.rename(os.path.abspath(os.path.join(dirpath, file)),
                    os.path.abspath(os.path.join(failed, file)))
            # The structure factor doesn't satify the Hasen-Verlet rule.
            elif np.max(test_case[:,-1]) > 2.8:
                os.rename(os.path.abspath(os.path.join(dirpath, file)),
                    os.path.abspath(os.path.join(failed, file)))
            else:
                os.rename(os.path.abspath(os.path.join(dirpath, file)),
                    os.path.abspath(os.path.join(passed, file)))


if __name__ == '__main__':
    directory = sys.argv[1]
    passed = sys.argv[2]
    failed = sys.argv[3]

    clean(directory, passed, failed)

