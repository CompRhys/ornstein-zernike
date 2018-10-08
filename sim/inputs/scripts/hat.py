import sys
import os 
import numpy as np 
# import matplotlib.pyplot as plt

# n_part      = [4096]
n_part      = [16384]

# System parameters
rho     = [0.3, 0.4, 0.5, 0.6]
force 	= [4.0, 6.0, 8.0, 10.0]
cutoff  = [2.0, 3.0] # don't make sigma too small 

test_number = 0
r_min       = 0.0
samples     = 4096

# plt.figure()

for p3 in force:
    for p4 in cutoff:
        for p1 in rho:
            for p9 in n_part:

                r_max  =  p4
                r = np.linspace(r_min,r_max,samples)

                potential = p3*(r-p4)*((r+p4)/(2*p4)-1.)
                force       = np.gradient(potential, r[1])
                potential   = potential + force[-1]*r_max
                potential   = potential - potential[-1] # shift potential
                force       = force - force[-1]
                
                # adaptive timestep setting.
                trunc = np.argmax(potential < 10)
                timestep = np.sqrt(0.5) / np.max((np.abs(force[trunc]),100.)) 
                # print(timestep)

                dictionary  = {'type':'tabulated','rho':p1,'min':r_min,'cutoff':r_max,
                                'timestep':timestep,'particles':p9}
                inpath      = os.path.expanduser('~')+'/Liquids/data/input/'
                np.save(inpath+'input_13'+format(test_number, "03")+'.npy', dictionary)

                tables      = np.vstack((potential, force))
                table_path  = os.path.expanduser('~')+'/Liquids/data/tables/'
                np.savetxt(table_path+'input_13'+format(test_number, "03")+'.dat', tables)

                test_number += 1

#                 plt.plot(r,potential)
#                 plt.plot(r,force)
#                 plt.ylim([-5,10]) 

# plt.show()