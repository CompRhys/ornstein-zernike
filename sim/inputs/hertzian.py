import sys
import os
import numpy as np
# import matplotlib.pyplot as plt

energy = [4.0, 6.0, 8.0, 10.0]
sigma = [2.0, 3.0]  # don't make sigma too small


test_number = 0
r_min = 0.0
samples = 4096

# plt.figure()

for p3 in energy:
    for p4 in sigma:

        r_max = p4
        r = np.linspace(r_min, r_max, samples)

        potential = p3 * np.power((1. - r / p4), 2.5)
        force = np.gradient(potential, r[1])
        potential = potential + force[-1] * r_max
        potential = potential - potential[-1]  # shift potential
        force = force - force[-1]

        tables = np.vstack((r, potential, force))
        table_path = os.path.expanduser('~') + '/closure/data/tables/'
        np.savetxt(table_path + 'input_14' +
                   format(test_number, "03") + '.dat', tables)

        test_number += 1

#                 plt.plot(r,potential)
#                 plt.plot(r,force)
#                 plt.ylim([-10,10])

# plt.show()
