import sys
import os
import re
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample, parse
import signal

# import matplotlib.pyplot as plt

def main(output_path):

    dt = 0.005
    temperature = 1.0

    burn_steps = 2048
    burn_iterations_max = 16

    sampling_iterations = 1024
    sampling_steps = 16

    box_size = 20

    density = 0.8
    
    # Control the density
    n_part = int(density * (box_size**3.))
    n_solv = n_part
    # box_size = np.power(n_part / rho, 1.0 / 3.0)
    print('Density={:.2f} \nNumber of Particles={} \nBox Size={:.1f}'
        .strip().format(density, n_part, box_size))

    # Box setup
    system = espressomd.System(box_l=[box_size] * 3)
    system.cell_system.skin = 0.2 * 3.0

    # Hardcode PRNG seed
    system.seed = 42
    np.random.seed()

    # Setup Real Particles
    for i in range(n_part):
        system.part.add(id=i, pos=np.random.random(3) * system.box_l, type=0)

    for i in range(n_solv):
        system.part.add(id=i+n_part, pos=np.random.random(3) * system.box_l, type=1)

    lj_eps = 1.0
    lj_sig = 1.0

    system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=lj_eps, sigma=lj_sig, cutoff=2.5, shift='auto')

    # Mixture of LJ particles

    system.non_bonded_inter[1, 0].lennard_jones.set_params(
    epsilon=lj_eps*1.1, sigma=lj_sig*0.75, cutoff=2.5, shift='auto')

    system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=lj_eps, sigma=lj_sig*0.5, cutoff=1.5, shift='auto')

    # Repulsive Smooth Step Solvent

    # system.non_bonded_inter[1, 0].smooth_step.set_params(
    #     d=0.7, n=10, eps=-1, k0=2.5, sig=1.5, cutoff=3,)

    # system.non_bonded_inter[1, 1].wca.set_params(
    # epsilon=lj_eps, sigma=lj_sig*0.6)


    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, dt, n_types=2)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, dt,
                                  temperature, burn_steps,
                                  burn_iterations_max)

    # Sample the RDF for the system
    rdf, r, sq, q, temp, t = sample.get_bulk(system, dt,
                                             sampling_iterations, 
                                             sampling_steps)

    # save rdf
    f_rdf = os.path.join(output_path,'rdf_ss_wca_p{}_b{}_t{}.dat'.format(
        density, box_size, temperature))

    if os.path.isfile(f_rdf):
        rdf_out = rdf
    else:
        rdf_out = np.vstack((r, rdf))

    with open(f_rdf, 'ab') as f:
        np.savetxt(f, rdf_out)

    # save sq
    f_sq = os.path.join(output_path,'sq_ss_wca_p{}_b{}_t{}.dat'.format(
        density, box_size, temperature))

    if os.path.isfile(f_sq):
        sq_out = sq
    else:
        sq_out = np.vstack((q, sq))

    with open(f_sq, 'ab') as f:
        np.savetxt(f, sq_out)


    # save temp
    f_temp = os.path.join(output_path,'temp_ss_wca_p{}_b{}_t{}.dat'.format(
        density, box_size, temperature))

    if os.path.isfile(f_temp):
        temp_old = np.loadtxt(f_temp)
        t_end = temp_old[0,-1]
        t += t_end + 1
    else:
        pass

    with open(f_temp, 'ab') as f:
        t_out = np.column_stack((t, temp))
        np.savetxt(f, t_out)


if __name__ == "__main__":
    output_path = sys.argv[1]
    main(output_path)
