import sys
import os 
import numpy as np 
# import matplotlib.pyplot as plt 

# n_part      = [4096]
n_part      = [16384]

# System parameters
rho         = [0.3, 0.4, 0.5, 0.6]
energy      = [4.0, 6.0, 8.0, 10.0]
inv_sigma   = [1.0, 1.5, 2.0]

test_number = 0
r_min       = 0.0
r_max       = 4.0 
samples     = 4096
r = np.linspace(r_min,r_max,samples)

# plt.figure()

for p3 in energy:
    for p2 in inv_sigma:
        for p1 in rho:
            for p9 in n_part:
                
                potential = p3*np.exp(-.5*np.power((r*p2),2))
                force       = np.gradient(potential, r[1])
                potential   = potential + force[-1]*r_max
                potential   = potential - potential[-1] # shift potential
                force       = force - force[-1] 

                # adaptive timestep setting.
                trunc = np.argmax(potential<10)
                timestep = np.sqrt(0.5) / np.max((np.abs(force[trunc]),100.)) 
                # print(timestep)

                dictionary  = {'type':'tabulated','rho':p1,'min':r_min,'cutoff':r_max,
                                'timestep':timestep,'particles':p9}
                inpath      = os.path.expanduser('~')+'/Liquids/data/input/'
                np.save(inpath+'input_12'+format(test_number, "03")+'.npy', dictionary)

                tables      = np.vstack((potential, force))
                table_path  = os.path.expanduser('~')+'/Liquids/data/tables/'
                np.savetxt(table_path+'input_12'+format(test_number, "03")+'.dat', tables)

                # plt.plot(r,potential)
                # plt.plot(r,force)
                # plt.ylim([-200,200]) 
                

                test_number += 1

# plt.show()
