import sys
import os
import re
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample
import matplotlib.pyplot as plt


def main(input_file):
    start = timeit.default_timer()

    # Simulation Parameters

    density = 0.7
    n_part = 512
    n_test = 1

    temperature = 1.
    timestep = 0.005

    burn_steps = 1024
    burn_iterations_max = 16

    sampling_steps = 8
    sampling_iterations = 8192   # choose 2**n for fp method
    dr = 0.02

    # Setup Espresso Environment
    system = setup.setup_box(input_file, density, n_part, n_test)

    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, timestep, n_test)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, timestep,
                                  temperature, burn_steps,
                                  burn_iterations_max)

    psi = sample.get_reference(system, sampling_iterations,
                            sampling_steps, n_part)

    # todo estimate the error in psi using DDA

    path = os.path.expanduser('~')
    output_path = path + '/masters/closure/data/raw/'
    test_number = re.findall('\d+', input_file)[0]

    np.savetxt(output_path + 'ref_' + test_number + '.dat', [psi[-1]])

    plt.figure()
    plt.plot(range(len(psi)), psi)
    plt.show()


if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
