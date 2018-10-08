import sys
import os 
import numpy as np 
# import matplotlib.pyplot as plt

# n_part      = [4096]
n_part      = [16384]

# System parameters
rho         = [0.3, 0.4, 0.5, 0.6]
alpha       = [2., 4., 6.]
kappa       = [2.5, 3.5]
delta       = 0.0

test_number = 0
r_min       = 0.0
samples     = 4096
r_max       = 3.0
r           = np.linspace(r_min,r_max,samples)
potential   = np.zeros(samples)

# plt.figure()


for p2 in alpha:
    for p3 in kappa:
        for p1 in rho:
            for p9 in n_part:

                potential[1:] = p2*np.exp(-p3*(r[1:]-delta))/r[1:]
                potential[0] = 2. * potential[1] - potential[2]

                force       = np.gradient(potential, r[1])
                potential   = potential + force[-1]*r_max
                potential   = potential - potential[-1] # shift potential
                force       = force - force[-1]
                
               # adaptive timestep setting.
                trunc = np.argmax(potential<50)
                timestep = np.sqrt(0.5) / np.max((np.abs(force[trunc]),100.)) 
                # print(timestep)

                dictionary  = {'type':'tabulated','rho':p1,'min':r_min,'cutoff':r_max,
                                'timestep':timestep,'particles':p9}
                inpath      = os.path.expanduser('~')+'/Liquids/data/input/'
                np.save(inpath+'input_6'+format(test_number, "03")+'.npy', dictionary)

                tables      = np.vstack((potential, force))
                table_path  = os.path.expanduser('~')+'/Liquids/data/tables/'
                np.savetxt(table_path+'input_6'+format(test_number, "03")+'.dat', tables)

                test_number += 1

#                 plt.plot(r,potential)
#                 plt.plot(r,force)
#                 plt.ylim([-100,100]) 
# plt.show()






