import os
import numpy as np
import matplotlib.pyplot as plt

# Notation

# read data 
path = os.path.expanduser('~')+'/closure/data/tables/'
# test_number = str(input("Input file number: "))
fig, axes = plt.subplots(1)

for test_number in [500,501,502,503]:
    phi = np.loadtxt(path+'input_'+str(test_number)+'.dat')

    axes.plot(phi[0,:], phi[1,:])

    axes.set_ylim([-3,6])

plt.show()
