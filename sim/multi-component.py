import sys
import os
import re
import espressomd
import timeit
import numpy as np
from core import setup, initialise, sample, parse, block


def main():
    
    start = timeit.default_timer()

    # Control the density of large particles
    box_l = np.power(n_part / rho, 1.0 / 3.0)
    print('Density={:.2f} \nNumber of Particles={} \nBox Size={:.1f}'
        .strip().format(rho, n_part, box_l))

    # Box setup
    system = espressomd.System(box_l=[box_l] * 3)

    # Hardcode PRNG seed
    system.seed = 42
    np.random.seed()

    # Setup Real Particles
    for i in range(n_part):
        system.part.add(id=i, pos=np.random.random(3) * syst.box_l, type=0)

    for i in range(n_solv):
        system.part.add(id=i+n_part, pos=np.random.random(3) * syst.box_l, type=1)

    system.cell_system.skin = 0.2 * r_cut_max



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
    test_number = re.findall('\d+', input_file)[-1]

    # save rdf
    f_rdf = 'rdf_test.dat'.format(
        output_path, density, n_part, temperature, test_number)

    if os.path.isfile(f_rdf):
        rdf_out = rdf
    else:
        rdf_out = np.vstack((r, rdf))

    with open(f_rdf, 'ab') as f:
        np.savetxt(f, rdf_out)

    # save sq
    f_sq = 'rq_test.dat'.format(
        output_path, density, n_part, temperature, test_number)

    if os.path.isfile(f_sq):
        sq_out = sq
    else:
        sq_out = np.vstack((q, sq))

    with open(f_sq, 'ab') as f:
        np.savetxt(f, sq_out)

    # save temp
    f_temp = 'temp_test.dat'.format(
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
    main()
