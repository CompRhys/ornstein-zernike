import sys
import os
import numpy as np
# import matplotlib.pyplot as plt

# System parameters
force = [4.0, 6.0, 8.0, 10.0]
cutoff = [2.0, 3.0]  # don't make sigma too small

test_number = 0
r_min = 0.0
samples = 4096

# plt.figure()

for p3 in force:
    for p4 in cutoff:

        r_max = p4
        r = np.linspace(r_min, r_max, samples)

        potential = p3 * (r - p4) * ((r + p4) / (2 * p4) - 1.)
        force = np.gradient(potential, r[1])
        potential = potential + force[-1] * r_max
        potential = potential - potential[-1]  # shift potential
        force = force - force[-1]

        tables = np.vstack((r, potential, force))
        table_path = os.path.expanduser('~') + '/closure/data/tables/'
        np.savetxt(table_path + 'input_13' +
                   format(test_number, "03") + '.dat', tables)

        test_number += 1

#                 plt.plot(r,potential)
#                 plt.plot(r,force)
#                 plt.ylim([-5,10])

# plt.show()
