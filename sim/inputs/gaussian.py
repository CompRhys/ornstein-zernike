import sys
import os
import numpy as np
# import matplotlib.pyplot as plt

energy = [4.0, 6.0, 8.0, 10.0]
inv_sigma = [1.0, 1.5, 2.0]

test_number = 0
r_min = 0.0
r_max = 4.0
samples = 4096
r = np.linspace(r_min, r_max, samples)

# plt.figure()

for p3 in energy:
    for p2 in inv_sigma:

        potential = p3 * np.exp(-.5 * np.power((r * p2), 2))
        force = np.gradient(potential, r[1])
        potential = potential + force[-1] * r_max
        potential = potential - potential[-1]  # shift potential
        force = force - force[-1]

        tables = np.vstack((r, potential, force))
        table_path = os.path.expanduser('~') + '/closure/data/tables/'
        np.savetxt(table_path + 'input_12' +
                   format(test_number, "03") + '.dat', tables)

        # plt.plot(r,potential)
        # plt.plot(r,force)
        # plt.ylim([-200,200])

        test_number += 1

# plt.show()
