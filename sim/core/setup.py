from __future__ import print_function
import espressomd
import numpy as np
import argparse


def setup_box(input_file, rho, n_part, n_test=0):
    """
    Initialises the system based off an input file
    """

    # Control the density
    box_l = np.power(n_part / rho, 1.0 / 3.0)
    print('Density={:.2f} \nNumber of Particles={} \nBox Size={:.1f}'
        .strip().format(rho, n_part, box_l))

    # Box setup
    syst = espressomd.System(box_l=[box_l] * 3)

    # Hardcode PRNG seed
    syst.seed = 42

    # Setup Real Particles
    for i in range(n_part - n_test):
        syst.part.add(id=i, pos=np.random.random(3) * syst.box_l, type=0)

    # Setup Pseudo Particles
    for i in range(n_test):
        syst.part.add(id=n_part-1-i, pos=np.random.random(3) * syst.box_l, type=n_test-i)


    set_potentials(syst, input_file, n_test)

    return syst


def bulk_potentials(system, input_file):
    """
    In order to be more consistent in out apporach and to facilitate adaptive time-step selection
    based off the hard shell repulsion all the potentials studied will be done via the tabulated 
    method. This is interesting as although we are introducing numerical erros with regard to the
    true analytical expressions we have used for inspiration when it comes to our purpose of
    deriving the best local closure deviations from the analytical inspirations do not matter 
    instead what we are interested in is ensuring that we have consistency across all of our
    measurements.
    """
    tables = np.loadtxt(input_file)

    system.force_cap = 0.  # Hard coded

    # skin depth; 0.2 is common choice see pg 556 Frenkel & Smit,
    system.cell_system.skin = 0.2 * tables[0, -1]

    system.non_bonded_inter[0, 0].tabulated.set_params(
        min=tables[0, 0], max=tables[0, -1],
        energy=tables[1, :], force=tables[2, :])

def set_potentials(system, input_file, n_test):
    """
    In order to be more consistent in out apporach and to facilitate adaptive time-step selection
    based off the hard shell repulsion all the potentials studied will be done via the tabulated 
    method. This is interesting as although we are introducing numerical erros with regard to the
    true analytical expressions we have used for inspiration when it comes to our purpose of
    deriving the best local closure deviations from the analytical inspirations do not matter 
    instead what we are interested in is ensuring that we have consistency across all of our
    measurements.
    """
    tables = np.loadtxt(input_file)

    system.force_cap = 0.  # Hard coded

    # skin depth; 0.2 is common choice see pg 556 Frenkel & Smit,
    system.cell_system.skin = 0.2 * tables[0, -1]

    system.non_bonded_inter[0, 0].tabulated.set_params(
        min=tables[0, 0], max=tables[0, -1],
        energy=tables[1, :], force=tables[2, :])

    for i in range(n_test):
        system.non_bonded_inter[n_test-i, 0].tabulated.set_params(
            min=tables[0, 0], max=tables[0, -1],
            energy=tables[1, :], force=tables[2, :])

    # system.non_bonded_inter[1, 1].tabulated.set_params(
    #     min=tables[0, 0], max=tables[0, -1],
    #     energy=np.zeros_like(tables[1, :]), force=np.zeros_like(tables[1, :]))