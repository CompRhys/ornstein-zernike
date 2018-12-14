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
         sampling_iterations, dr, r_size_max, output_path):
    """
    Evaluate the chemical potential and the cavity function

    returns:
        cav_out = (r,cav) array(iterations+1,bins)
        mu = (mu) array(iterations)
    """
    start = timeit.default_timer()

    sampling_iterations = 2
    sampling_steps = 128

    # Setup Espresso Environment
    system = setup.setup_box(input_file, density, n_part)

    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, timestep)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, timestep,
                                  temperature, burn_steps,
                                  sampling_steps,
                                  burn_iterations_max)

    bins = np.ceil(r_size_max / dr).astype(int) + 1
    # r = np.arange(0, dr * (bins), dr)
    r = np.arange(0, dr * (bins), dr)

    cav, mu = sample.sample_cavity(system, timestep, sampling_iterations,
                                   sampling_steps, n_part, r, dr, bins)

    # save the results
    test_number = re.findall('\d+', input_file)[0]

    # save cavity central potentials
    f_cav = '{}cav_d{}_n{}_t{}_p{}.dat'.format(output_path, density, n_part,
                                               temperature, test_number)

    if os.path.isfile(f_cav):
        cav_out = cav
    else:
        cav_out = np.vstack((r, cav))

    with open(f_cav, 'ab') as f:
        np.savetxt(f, cav_out)

    # save chemical potential
    f_mu = '{}mu_d{}_n{}_t{}_p{}.dat'.format(output_path, density, n_part,
                                             temperature, test_number)

    with open(f_mu, 'ab') as f:
        np.savetxt(f, mu)

    print(timeit.default_timer() - start)


if __name__ == "__main__":
    opt = parse.parse_input()

    main(opt.table, opt.cav_part, opt.rho, opt.temp, opt.dt,
         opt.burn_steps, opt.burn_iter_max, opt.timesteps,
         opt.cav_iter, opt.dr, opt.r_cav, opt.output)
