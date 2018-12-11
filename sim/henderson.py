import sys
import os
import re
import argparse
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample, parse
import matplotlib.pyplot as plt


def main(input_file, n_part, density, temperature, timestep,
         burn_steps, burn_iterations_max, sampling_steps,
         sampling_iterations, dr, r_size_max):
    start = timeit.default_timer()

    # Setup Espresso Environment
    system = setup.setup_box(input_file, density, n_part)

    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, timestep)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, timestep,
                                  temperature, burn_steps,
                                  burn_iterations_max)

    # hard coded
    r_size_max = 1.2
    bins = np.ceil(r_size_max / dr).astype(int)
    r = np.arange(dr, dr * (bins + 1), dr)

    cav, mu = sample.sample_cavity(system, timestep, sampling_iterations, sampling_steps,
                                   n_part, dr, bins)

    print('Mean Chemical Potential {} +/- {}'.format(np.mean(mu), np.std(mu)))

    path = os.path.expanduser('~')
    output_path = path + '/masters/closure/data/raw/'
    test_number = re.findall('\d+', input_file)[0]

    cav_out = np.column_stack((r, cav.T))
    np.savetxt('{}cav_d{}_n{}_p{}.dat'
               .format(output_path, density, n_part, test_number), cav_out)
    np.savetxt('{}mu_d{}_n{}_p{}.dat'
               .format(output_path, density, n_part, test_number), mu)

    print(timeit.default_timer() - start)

    display_figures = True 
    if display_figures:
        plot_figures(r, cav, mu)


def plot_figures(r, cav, mu):
    fig, axes = plt.subplots(2,2, figsize=(10,6))

    axes[0,0] = plt.plot(r, cav.T/mu)    
    axes[0,1] = plt.plot(r, cav.T/np.mean(mu))    
    axes[1,0] = plt.hist(mu, 'auto')    
    axes[1,1] = plt.plot(r, np.mean(cav, axis = 0)/np.mean(mu))

    fig.tight_layout()
    plt.show    

if __name__ == "__main__":
    opt = parse.parse_input()

    main(opt.table, opt.cav_part, opt.rho, opt.temp, opt.dt,
         opt.burn_steps, opt.burn_iter_max, opt.timesteps,
         opt.cav_iter, opt.dr, opt.r_cav)
