import sys
import os
import re
import argparse
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample, parse


def main(input_file, density, temperature, dr, dt, 
        burn_steps, burn_iterations_max, n_part, sampling_steps,
        sampling_iterations,  output_path):
    start = timeit.default_timer()

    # Setup Espresso Environment
    system = setup.setup_box(input_file, density, n_part)

    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, dt)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, dt,
                                  temperature, burn_steps,
                                  burn_iterations_max)

    # order of 8 / (2*pi / L) means we take s(q) up to q = 8 i.e. past
    # principle peak
    sq_order = np.ceil(4 * system.box_l[0] / np.pi).astype(int)

    # Sample the RDF for the system
    rdf, r, sq, q, temp, t = sample.get_bulk(system, dt,
                                             sampling_iterations, dr,
                                             sq_order, sampling_steps)

    # Extract the interaction potential used by the model
    phi = sample.get_phi(system, r)

    # save the results
    test_number = re.findall('\d+', input_file)[-1]

    # save rdf
    f_rdf = '{}rdf_d{}_n{}_t{}_p{}.dat'.format(
        output_path, density, n_part, temperature, test_number)

    if os.path.isfile(f_rdf):
        rdf_out = rdf
    else:
        rdf_out = np.vstack((r, rdf))

    with open(f_rdf, 'ab') as f:
        np.savetxt(f, rdf_out)

    # save sq
    f_sq = '{}sq_d{}_n{}_t{}_p{}.dat'.format(
        output_path, density, n_part, temperature, test_number)

    if os.path.isfile(f_sq):
        sq_out = sq
    else:
        sq_out = np.vstack((q, sq))

    with open(f_sq, 'ab') as f:
        np.savetxt(f, sq_out)

    # save phi
    f_phi = '{}phi_p{}.dat'.format(
        output_path, test_number)

    if os.path.isfile(f_phi):
        pass
    else:
        phi_out = np.vstack((r, phi))
        np.savetxt(f_phi, phi_out)

    # save temp
    f_temp = '{}temp_d{}_n{}_t{}_p{}.dat'.format(
        output_path, density, n_part, temperature, test_number)

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
        opt.bulk_part, 
        opt.bulk_steps,
        opt.bulk_iter, 
        opt.output)
