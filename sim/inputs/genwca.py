import sys
import os
import numpy as np
# import matplotlib.pyplot as plt

# System parameters
energy = [1.0, 0.6]
powers = [[50, 49], [35, 22], [20, 14], [12, 6]]

test_number = 0
r_min = 0.0
samples = 4096

for i in np.arange(len(powers)):

    alpha = powers[i][0]
    beta = powers[i][1]
    r_max = np.power(alpha / beta, 1 / (alpha - beta))
    r = np.linspace(r_min, r_max, samples)
    rs = np.zeros_like(r)
    rh = np.zeros_like(r)
    rs[1:] = np.power(1. / r[1:], beta)
    rs[0] = 2 * rs[1] - rs[2]
    rh[1:] = rs[1:] * np.power(1 / r[1:], alpha - beta)
    rh[0] = 2 * rh[1] - rh[2]
    prefactor = alpha * np.power(alpha / beta, beta)

    for p2 in energy:

        potential = prefactor * p2 * (rh - rs) + p2
        force = np.gradient(potential, r[1])
        potential = potential + force[-1] * r_max
        potential = potential - potential[-1]  # shift potential
        force = force - force[-1]

        tables = np.vstack((r, potential, force))
        table_path = os.path.expanduser('~') + '/closure/data/tables/'
        np.savetxt(table_path + 'input_4' +
                   format(test_number, "03") + '.dat', tables)

        test_number += 1

#                 plt.plot(r,potential)
#                 plt.plot(r,force)
#                 plt.ylim([-10,100])

# plt.show()
