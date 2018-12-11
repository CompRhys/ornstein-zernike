import sys
import os
import re
import argparse
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample, parse


def main(input_file, n_part, density, temperature, timestep, 
        burn_steps, burn_iterations_max, sampling_steps,
        sampling_iterations, dr):
    start = timeit.default_timer()

    # Setup Espresso Environment
    system = setup.setup_box(input_file, density, n_part)

    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, timestep)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, timestep,
                                  temperature, burn_steps,
                                  burn_iterations_max)

    # order of 8 / (2*pi / L) means we take s(q) up to q = 8 i.e. past
    # principle peak
    sq_order = np.ceil(4 * system.box_l[0] / np.pi).astype(int)

    # Sample the RDF for the system
    rdf, r, sq, q, kinetic_temp, t = sample.get_bulk(system, timestep,
                                                     sampling_iterations, dr,
                                                     sq_order, sampling_steps)

    # Extract the interaction potential used by the model
    phi = sample.get_phi(system, r)

    dat_out = np.array([density, system.box_l[0], len(r), n_part])
    rdf_out = np.column_stack((r, rdf.T))
    sq_out = np.column_stack((q, sq.T))
    phi_out = np.column_stack((r, phi))
    temp_out = np.column_stack((t, kinetic_temp))

    path = os.path.expanduser('~')
    output_path = path + '/masters/closure/data/raw/'

    test_number = re.findall('\d+', input_file)[0]

    np.savetxt('{}dat_d{}_n{}_p{}.dat'
               .format(output_path, density, n_part, test_number), dat_out)
    np.savetxt('{}rdf_d{}_n{}_p{}.dat'
               .format(output_path, density, n_part, test_number), rdf_out)
    np.savetxt('{}sq_d{}_n{}_p{}.dat'
               .format(output_path, density, n_part, test_number), sq_out)
    np.savetxt('{}phi_d{}_n{}_p{}.dat'
               .format(output_path, density, n_part, test_number), phi_out)
    np.savetxt('{}temp_d{}_n{}_p{}.dat'
               .format(output_path, density, n_part, test_number), temp_out)

    print(timeit.default_timer() - start)


if __name__ == "__main__":
    input_file = sys.argv[1]

    opt = parse.parse_input()

    main(opt.table, opt.bulk_part, opt.rho, opt.temp, opt.dt, 
        opt.burn_steps, opt.burn_iter_max, opt.timesteps,
        opt.bulk_iter, opt.dr)
