import sys
import os
import re
import argparse
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample, parse, block


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

    # Sample the RDF for the system
    pos, vel, temp, t = sample.get_pos_vel(system, dt,
                                             sampling_iterations,
                                             sampling_steps)

    # save the results
    test_number = re.findall('\d+', input_file)[-1]

    # save positions
    f_pos = '{}pos_d{}_n{}_t{}_p{}.dat'.format(
        output_path, density, n_part, temperature, test_number)

    with open(f_pos, 'ab') as f:
        np.savetxt(f, pos)

    # save velocities
    f_vel = '{}vel_d{}_n{}_t{}_p{}.dat'.format(
        output_path, density, n_part, temperature, test_number)

    with open(f_vel, 'ab') as f:
        np.savetxt(f, vel)

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
