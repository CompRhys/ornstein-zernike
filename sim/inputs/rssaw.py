import sys
import os
import numpy as np
# import matplotlib.pyplot as plt

sigma1 = [0.8, 1.15, 1.5]
sigma2 = [1.0, 1.35]
lamda1 = [0.5]
lamda2 = [0.3]

test_number = 0
r_min = 0.0
r_max = 3.0
samples = 4096
r = np.linspace(r_min, r_max, samples)

# plt.figure()

for p2 in lamda1:
    for p4 in lamda2:
        for p5 in sigma1:
            for p6 in sigma2:

                    phi_hard = np.zeros(samples)
                    phi_hard[1:] = np.power((1. / r[1:]), 14)
                    phi_hard[0] = 2. * phi_hard[1] - phi_hard[2]

                    phi_step = p2 * np.tanh(10 * (r - p5))

                    phi_well = p4 * np.tanh(10 * (r - p6))

                    potential = phi_step + phi_hard - phi_well
                    force = np.gradient(potential, r[1])
                    potential = potential + force[-1] * r_max
                    potential = potential - potential[-1]  # shift potential
                    force = force - force[-1]

                    tables = np.vstack((r, potential, force))
                    table_path = os.path.expanduser(
                        '~') + '/closure/data/tables/'
                    np.savetxt(table_path + 'input_15' +
                               format(test_number, "03") + '.dat', tables)

                    test_number += 1

#                             plt.plot(r,potential)
#                             # plt.plot(r,force)
#                             plt.ylim([-5,10])
#                             # plt.xlim([1,2])
# plt.show()
