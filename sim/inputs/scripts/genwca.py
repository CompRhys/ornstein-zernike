import sys
import os 
import numpy as np 
# import matplotlib.pyplot as plt

n_part      = [16384]
# n_part      = [4096]

# System parameters
# rho         = np.arange(0.3, 0.45, 0.025).tolist()
rho         = [0.3, 0.4, 0.5, 0.6]
energy      = [1.0, 0.6]
powers = [[50,49], [35,22], [20,14], [12,6]]
# powers = [[20,14]]


test_number = 0
r_min       = 0.0
samples     = 4096



# plt.figure()
for i in np.arange(len(powers)):
    
    alpha      = powers[i][0]
    beta       = powers[i][1]
    r_max       = np.power(alpha/beta, 1/(alpha-beta))
    r           = np.linspace(r_min,r_max,samples)
    rs         = np.zeros_like(r)
    rh         = np.zeros_like(r)
    rs[1:]     = np.power(1./r[1:], beta)
    rs[0]      = 2*rs[1]-rs[2]
    rh[1:]     = rs[1:]*np.power(1/r[1:], alpha-beta) 
    rh[0]      = 2*rh[1]-rh[2]
    prefactor = alpha*np.power(alpha/beta, beta)

    for p2 in energy:
        for p1 in rho:
            for p9 in n_part:

                potential   = prefactor*p2*(rh-rs) + p2
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
                np.save(inpath+'input_4'+format(test_number, "03")+'.npy', dictionary)

                tables      = np.vstack((potential, force))
                table_path  = os.path.expanduser('~')+'/Liquids/data/tables/'
                np.savetxt(table_path+'input_4'+format(test_number, "03")+'.dat', tables)

                test_number += 1

#                 plt.plot(r,potential)
#                 plt.plot(r,force)
#                 plt.ylim([-10,100]) 

# plt.show()