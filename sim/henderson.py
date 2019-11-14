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
         sampling_iterations, r_size_max, mu_repeats, output_path):
    """
    Evaluate the chemical potential and the cavity function

    returns:
        cav_out = (r,cav) array(iterations+1,bins)
        mu = (mu) array(iterations)
    """
    start = timeit.default_timer()

    # Setup Espresso Environment
    system = setup.setup_box(input_file, density, n_part)

    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, dt)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, dt,
                                  temperature, burn_steps,
                                  burn_iterations_max)

    bins = np.ceil(r_size_max / dr).astype(int) + 1
    # r = np.arange(0, dr * (bins), dr)
    r = (np.arange(bins) + 1) * dr

    cav, mu = sample.sample_henderson(system, dt, sampling_iterations,
                                      sampling_steps, n_part, mu_repeats,
                                      r, dr, bins, input_file)

    

    # save the results
    test_number = re.findall('\d+', input_file)[-1]
    print(test_number)

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

    main(opt.table,
         opt.rho,
         opt.temp,
         opt.dr,
         opt.dt,
         opt.burn_steps,
         opt.burn_iter_max,
         opt.cav_part,
         opt.cav_steps,
         opt.cav_iter,
         opt.cav_radius,
         opt.mu_repeats,
         opt.output)
