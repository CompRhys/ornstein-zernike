import sys
import os 
import numpy as np 
# import matplotlib.pyplot as plt

# n_part      = [4096]
n_part      = [16384]

# System parameters
# rho         = [0.6]
rho         = [0.3, 0.4, 0.5, 0.6]

# Interaction parameters (Soft Sphere)

diameter    = [.7]
exponent    = 12.
energy      = [1.0]
# kappa       = [1.5]
kappa       = [1.0, 3.0, 5.0, 7.0]
# sigma       = [2.0]
sigma       = [1.0, 1.5, 2.0]
cutoff      = 0.001

test_number = 0
samples     = 4096
r_min       = 0.0
r_max       = 3.0
r           = np.linspace(r_min,r_max,samples)




# plt.figure()

for p2 in energy:
    for p4 in sigma:
        for p5 in kappa:
            for p1 in rho:
                for p9 in n_part:

                    # r_max       =  p4+(0.5*np.log((p2-cutoff)/cutoff))/p5
                    # r           = np.linspace(r_min,r_max,samples)

                    phi_step    = p2 / (1. + np.exp(2*p5*(r-p4)))

                    phi_hard    = np.zeros(samples)
                    phi_hard[1:] = np.power((1./r[1:]),exponent)
                    phi_hard[0] = 2. * phi_hard[1] - phi_hard[2]
                    
                    potential = phi_step + phi_hard

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
                    np.save(inpath+'input_8'+format(test_number, "03")+'.npy', dictionary)

                    tables      = np.vstack((potential, force))
                    table_path  = os.path.expanduser('~')+'/Liquids/data/tables/'
                    np.savetxt(table_path+'input_8'+format(test_number, "03")+'.dat', tables)

                    test_number += 1

#                     plt.plot(r,potential)
#                     plt.plot(r,force)
#                     plt.ylim([-5,10]) 
#                     # plt.xlim([1,2]) 
# plt.show()






