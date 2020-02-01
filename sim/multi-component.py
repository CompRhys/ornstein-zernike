import sys
import os
import re
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample, parse

# import matplotlib.pyplot as plt


def main():

    dt = 0.005
    temperature = 1.0

    burn_steps = 2048
    burn_iterations_max = 16

    sampling_iterations = 1024
    sampling_steps = 16

    n_part = 512
    n_solv = 512

    density = 0.8
    
    start = timeit.default_timer()

    # Control the density of large particles
    box_l = np.power(n_part / density, 1.0 / 3.0)
    print('Density={:.2f} \nNumber of Particles={} \nBox Size={:.1f}'
        .strip().format(density, n_part, box_l))

    # Box setup
    system = espressomd.System(box_l=[box_l] * 3)

    # Hardcode PRNG seed
    system.seed = 42
    np.random.seed()

    # Setup Real Particles
    for i in range(n_part):
        system.part.add(id=i, pos=np.random.random(3) * system.box_l, type=0)

    system.cell_system.skin = 0.2 * 3.0

    lj_eps = 1.0
    lj_sig = 1.0

    system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=lj_eps, sigma=lj_sig, cutoff=2.5, shift='auto')

    # # Disperse Particles to energy minimum
    # initialise.disperse_energy(system, temperature, dt, n_types=1)

    # # Integrate the system to warm up to specified temperature
    # initialise.equilibrate_system(system, dt,
    #                               temperature, burn_steps,
    #                               burn_iterations_max)

    # # Sample the RDF for the system
    # rdf, r, sq, q, temp, t = sample.get_bulk(system, dt,
    #                                          sampling_iterations, 
    #                                          sampling_steps)

    for i in range(n_solv):
        system.part.add(id=i+n_part, pos=np.random.random(3) * system.box_l, type=1)


    # Mixture of LJ particles

    system.non_bonded_inter[1, 0].lennard_jones.set_params(
    epsilon=lj_eps*1.1, sigma=lj_sig*0.75, cutoff=2.5, shift='auto')

    system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=lj_eps, sigma=lj_sig*0.5, cutoff=1.5, shift='auto')

    # Repulsive Smooth Step Solvent

    # system.non_bonded_inter[1, 0].smooth_step.set_params(
    #     d=0.7, n=10, eps=-1, k0=2.5, sig=1.5, cutoff=3,)

    # system.non_bonded_inter[1, 1].wca.set_params(
    # epsilon=wca_eps, sigma=wca_sig*0.6)

    # Disperse Particles to energy minimum
    initialise.disperse_energy(system, temperature, dt, n_types=2)

    # Integrate the system to warm up to specified temperature
    initialise.equilibrate_system(system, dt,
                                  temperature, burn_steps,
                                  burn_iterations_max)

    # Sample the RDF for the system
    rdf_ex, r_ex, sq_ex, q_ex, temp_ex, t_ex = sample.get_bulk(system, dt,
                                             sampling_iterations, 
                                             sampling_steps)


    # plt.figure()
    # plt.plot(r_ex, np.mean(rdf_ex, axis=0))

    # plt.figure()
    # plt.plot(q_ex, np.mean(sq_ex, axis=0))
    # plt.show()

    # # save rdf
    # f_rdf = 'rdf_test.dat'

    # if os.path.isfile(f_rdf):
    #     rdf_out = rdf
    # else:
    #     rdf_out = np.vstack((r, rdf))

    # with open(f_rdf, 'ab') as f:
    #     np.savetxt(f, rdf_out)

    # # save sq
    # f_sq = 'rq_test.dat'

    # if os.path.isfile(f_sq):
    #     sq_out = sq
    # else:
    #     sq_out = np.vstack((q, sq))

    # with open(f_sq, 'ab') as f:
    #     np.savetxt(f, sq_out)

    # # save temp
    # f_temp = 'temp_test.dat'

    # if os.path.isfile(f_temp):
    #     temp_old = np.loadtxt(f_temp)
    #     t_end = temp_old[0,-1]
    #     t += t_end + 1
    # else:
    #     pass

    # with open(f_temp, 'ab') as f:
    #     t_out = np.column_stack((t, temp))
    #     np.savetxt(f, t_out)

    print(timeit.default_timer() - start)


if __name__ == "__main__":
    main()
