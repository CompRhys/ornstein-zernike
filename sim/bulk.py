import sys
import os
import re
import espressomd
import timeit
import numpy as np
from core import liquids


def main(input_file):
    start = timeit.default_timer()

    # Simulation Parameters
    density = 0.6
    n_part = 4096

    temperature = 1.
    timestep = 0.005

    burn_steps = 1024
    burn_iterations_max = 16

    sampling_steps = 8
    sampling_iterations = 1024    # choose 2**n for fp method
    dr = 0.025

    # Setup Espresso Environment
    system = liquids.initialise_system(input_file, density, n_part)

    # Disperse Particles to energy minimum
    min_dist = liquids.disperse_energy(system, temperature, timestep)

    # Integrate the system to warm up to specified temperature
    liquids.equilibrate_system(system, timestep,
                               temperature, burn_steps,
                               burn_iterations_max)

    # 8 / (2*pi / L) means we take s(q) up to q = 8 i.e. past principle peak
    sq_order = np.ceil(4 * system.box_l[0] / np.pi).astype(int)

    # Sample the RDF for the system
    rdf, r, sq, q, kinetic_temp, t = liquids.sample_combo(system, timestep, sampling_iterations,
                                                          dr, sq_order, sampling_steps)

    # Extract the interaction potential used by the model
    phi = liquids.sample_phi(system, r)

    dat_out = np.array([density,system.box_l[0],len(r), n_part])
    rdf_out = np.column_stack((r, rdf.T))
    sq_out = np.column_stack((q, sq.T))
    phi_out = np.column_stack((r, phi))
    temp_out = np.column_stack((t, kinetic_temp))

    path = os.path.expanduser('~')
    output_path = path + '/masters/closure/data/raw/'

    test_number = re.findall('\d+', input_file)[0]

    np.savetxt(output_path + 'dat_' + test_number + '.dat', dat_out)
    np.savetxt(output_path + 'rdf_' + test_number + '.dat', rdf_out)
    np.savetxt(output_path + 'sq_' + test_number + '.dat', sq_out)
    np.savetxt(output_path + 'phi_' + test_number + '.dat', phi_out)
    np.savetxt(output_path + 'temp_' + test_number + '.dat', temp_out)


if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
