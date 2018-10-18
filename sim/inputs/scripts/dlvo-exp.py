import sys
import os 
import numpy as np
# import matplotlib.pyplot as plt 

# System parameters
rho     = [0.3, 0.4, 0.5, 0.6]
energy  = [3.0, 5.0, 7.0] # 0.5 - 11.0
kappa   = [10., 20., 30.] # 0.5 - 30.0
# kappa   = np.arange(0.5, 30.0, 0.5).tolist()
shift   = 0.3

test_number = 0
r_min       = 0.000
r_max       = 3.0
samples     = 4096
r           = np.linspace(r_min,r_max,samples)
r4          = np.power((1./(r+shift)), 4.)
r4[0]       = 2*r4[1] - r4[2]
r8          = np.square(r4)
r12         = np.power(r4, 3.)

# plt.figure()

for p3 in energy:
    for p5 in kappa:
        for p1 in rho:
            for p9 in n_part:

                rexp = np.exp(-p5*np.power((r-1.+shift),4))/(r+shift)
                rexp[0] = 2*rexp[1]-rexp[2]

                potential = p3*(r12-r8)+rexp
                force       = np.gradient(potential, r[1])
                potential   = potential + force[-1]*r_max
                potential   = potential - potential[-1] # shift potential
                force       = force - force[-1]

                # adaptive timestep setting.
                trunc = np.argmax(potential<10)
                timestep = np.sqrt(0.5) / np.max((np.abs(force[trunc]),100.)) 
                # print(timestep)

                dictionary  = {'rho':p1,'min':r_min,'cutoff':r_max}
                inpath      = os.path.expanduser('~')+'/Liquids/data/input/'
                np.save(inpath+'input_11'+format(test_number, "03")+'.npy', dictionary)

                tables      = np.vstack((potential, force))
                table_path  = os.path.expanduser('~')+'/Liquids/data/tables/'
                np.savetxt(table_path+'input_11'+format(test_number, "03")+'.dat', tables)

                test_number += 1

#                 plt.plot(r,potential)
#                 plt.plot(r,force)
#                 plt.ylim([-1,5]) 
#                 plt.xlim([0,5])
# plt.show()
