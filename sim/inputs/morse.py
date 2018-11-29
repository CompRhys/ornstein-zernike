import sys
import os
import numpy as np
# import matplotlib.pyplot as plt

# System parameters
energy = [0.65, 0.6, 0.55]
minimum = [1.0]
alpha = [5.0, 9.0]


test_number = 0
r_min = 0.0
samples = 4096
r_max = 3.0
r = np.linspace(r_min, r_max, samples)

# plt.figure()

for p2 in energy:
    for p3 in minimum:
        for p4 in alpha:

            exp1 = np.exp(-2. * p4 * (r - p3))
            exp2 = np.exp(-p4 * (r - p3))

            potential = p2 * (exp1 - 2. * exp2)

            force = np.gradient(potential, r[1])
            potential = potential + force[-1] * r_max
            potential = potential - potential[-1]  # shift potential
            force = force - force[-1]

            tables = np.vstack((r, potential, force))
            table_path = os.path.expanduser(
                '~') + '/closure/data/tables/'
            np.savetxt(table_path + 'input_3' +
                       format(test_number, "03") + '.dat', tables)

            test_number += 1

#                     plt.plot(r,potential)
#                     plt.plot(r,force)
#                     plt.ylim([-5,5])
#                     plt.xlim([0,5])

# plt.show()
