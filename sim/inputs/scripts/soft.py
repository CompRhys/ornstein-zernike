import os 
import numpy as np 
# import matplotlib.pyplot as plt

# n_part      = [4096]
n_part      = [16384]

# System parameters
rho         = [0.3, 0.4, 0.5, 0.6]

# Interaction parameters

energy      = [1.0, 0.6]
exponent    = [4.0, 6.0, 8.0, 10.0]

test_number = 0
r_min       = 0.0
r_max       = 3.0
samples     = 4096
r           = np.linspace(r_min,r_max,samples)

# plt.figure()  

for p3 in energy:
    for p5 in exponent:
        for p1 in rho:
            for p9 in n_part:

                potential = np.zeros(samples)
                potential[1:] = p3*(1./r[1:])**p5
                potential[0] = 2* potential[1] - potential[2]

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
                np.save(inpath+'input_2'+format(test_number, "03")+'.npy', dictionary)

                tables      = np.vstack((potential, force))
                table_path  = os.path.expanduser('~')+'/Liquids/data/tables/'
                np.savetxt(table_path+'input_2'+format(test_number, "03")+'.dat', tables)

          
                # plt.plot(r,potential)
                # plt.plot(r,force)
                # plt.ylim([-200,200]) 

                test_number += 1

# plt.show()




