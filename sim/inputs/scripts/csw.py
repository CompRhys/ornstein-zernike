import sys
import os 
import numpy as np 
# import matplotlib.pyplot as plt

# n_part      = [4096]
n_part      = [16384]

# System parameters

# rho         = [0.4]
rho         = [0.3, 0.4, 0.5, 0.6]

# Interaction parameters (Soft Sphere)


kappa       = [15.0, 5.0] 
offset_r    = [1.2, 1.6] # offset_r is scaling factor from sigma == 1
energy_r    = [1.] 
energy_a    = [2.0] 
delta_a     = [0.1, 0.2] # delta_a is scaling factor from sigma
offset_a    = [2.] # offset_a is scaling factor from sigma

test_number = 0 
r_min       = 0.0
r_max       = 3.0
samples     = 4096
r           = np.linspace(r_min,r_max,samples)

# plt.figure()

for p2 in energy_a:
    for p4 in kappa:
        for p5 in offset_r:
            for p6 in energy_r:
                for p7 in delta_a:
                    for p8 in offset_a:
                        for p1 in rho:
                            for p9 in n_part:

                                    phi_hard        = np.zeros(samples)
                                    phi_hard[1:]    = np.power((1./r[1:]),12)
                                    phi_hard[0]     = 2. * phi_hard[1] - phi_hard[2]

                                    phi_step    = p2 / (1. + np.exp(p4*(r-p5)))

                                    phi_gauss   = p6 * np.exp(-0.5*np.square((r-p8)/(p7)))

                                    potential   = phi_step + phi_hard - phi_gauss
                                    force       = np.gradient(potential, r[1])
                                    potential   = potential + force[-1]*r_max
                                    potential   = potential - potential[-1] # shift potential
                                    force       = force - force[-1]


                                    # adaptive timestep setting.
                                    trunc       = np.argmax(potential<10)
                                    timestep    = np.sqrt(0.5) / np.max((np.abs(force[trunc]),100.)) 
                                    # print(timestep)

                                    dictionary  = {'type':'tabulated','rho':p1,'min':r_min,'cutoff':r_max,
                                                    'timestep':timestep,'particles':p9, }

                                    inpath      = os.path.expanduser('~')+'/Liquids/data/input/'
                                    np.save(inpath+'input_9'+format(test_number, "03")+'.npy', dictionary)

                                    tables      = np.vstack((potential, force))
                                    table_path  = os.path.expanduser('~')+'/Liquids/data/tables/'
                                    np.savetxt(table_path+'input_9'+format(test_number, "03")+'.dat', tables)

                                    test_number += 1

#                                     plt.plot(r,potential)
#                                     plt.plot(r,force)
#                                     plt.ylim([-5,10]) 
#                                     # plt.xlim([1,2]) 
# plt.show()






