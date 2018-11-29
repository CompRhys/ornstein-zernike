import os
import numpy as np

table_path = os.path.expanduser('~') + '/masters/closure/data/tables/'

# System parameters
energy = [0.65, 0.6, 0.55, 0.5]

test_number = 0
r_min = 0.0
r_max = 2.0
samples = 4096
r = np.linspace(r_min, r_max, samples)
r6 = np.zeros_like(r)
r6[1:] = np.power(1. / r[1:], 6.)
r6[0] = 2 * r6[1] - r6[2]
r12 = np.power(r6, 2.)

# plt.figure()

for p2 in energy:

    potential = 4 * p2 * (r12 - r6)
    force = np.gradient(potential, r[1])
    potential = potential + force[-1] * r_max
    potential = potential - potential[-1]  # shift potential
    force = force - force[-1]

    tables = np.vstack((r, potential, force))

    np.savetxt(table_path + 'input_1' + format(test_number, "03") + '.dat', tables)

    test_number += 1


