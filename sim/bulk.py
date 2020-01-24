import sys
import os
import re
import argparse
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample, parse


def main(input_file, density, temperature, dr, dt, 
        burn_steps, burn_iterations_max, box_size, sampling_steps,
        sampling_iterations,  output_path):
    start = timeit.default_timer()

    # Setup Espresso Environment
    system, n_part = setup.setup_box(input_file, density, box_size)

    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, dt)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, dt,
                                  temperature, burn_steps,
                                  burn_iterations_max)

    # Sample the RDF for the system
    rdf, r, sq, q, temp, t = sample.get_bulk(system, dt,
                                             sampling_iterations, 
                                             sampling_steps)

    # Extract the interaction potential used by the model
    phi = sample.get_phi(system, r)

    # save the results
    _, pot_type, pot_number = input_file.split("_")
    pot_number = re.findall('\d+', pot_number)[-1]

    print(pot_number, pot_type)

    # save rdf
    f_rdf = '{}rdf_{}_{}_p{}_n{}_t{}.dat'.format(
        output_path, pot_type, pot_number, density, n_part, temperature)

    if os.path.isfile(f_rdf):
        rdf_out = rdf
    else:
        rdf_out = np.vstack((r, rdf))

    with open(f_rdf, 'ab') as f:
        np.savetxt(f, rdf_out)

    # save sq
    f_sq = '{}sq_{}_{}_p{}_n{}_t{}.dat'.format(
        output_path, pot_type, pot_number, density, n_part, temperature)

    if os.path.isfile(f_sq):
        sq_out = sq
    else:
        sq_out = np.vstack((q, sq))

    with open(f_sq, 'ab') as f:
        np.savetxt(f, sq_out)

    # save phi
    f_phi = '{}phi_{}_{}.dat'.format(
        output_path, pot_type, pot_number)

    if os.path.isfile(f_phi):
        pass
    else:
        phi_out = np.vstack((r, phi))
        np.savetxt(f_phi, phi_out)

    # save temp
    f_temp = '{}temp_{}_{}_p{}_n{}_t{}.dat'.format(
        output_path, pot_type, pot_number, density, n_part, temperature)

    if os.path.isfile(f_temp):
        temp_old = np.loadtxt(f_temp)
        t_end = temp_old[0,-1]
        t += t_end + 1
    else:
        pass

    with open(f_temp, 'ab') as f:
        t_out = np.column_stack((t, temp))
        np.savetxt(f, t_out)

    print(timeit.default_timer() - start)


if __name__ == "__main__":
    opt = parse.parse_input()

    main(opt.table, 
        opt.rho, 
        opt.temp, 
        opt.dr, 
        opt.dt, 
        opt.burn_steps, 
        opt.burn_iter_max, 
        opt.box_size, 
        opt.bulk_steps,
        opt.bulk_iter, 
        opt.output)
