from __future__ import print_function
import espressomd
import numpy as np
import argparse


def setup_box(input_file, rho, box_l, n_test=0):
    """
    Initialises the system based off an input file
    """

    # Control the density
    n_part = int(rho * (box_l**3.))
    # box_l = np.power(n_part / rho, 1.0 / 3.0)
    print('Potential={} \nDensity={:.2f} \nNumber of Particles={} \nBox Size={:.1f}'
        .strip().format(input_file, rho, n_part, box_l))

    # Box setup
    syst = espressomd.System(box_l=[box_l] * 3)

    # Hardcode PRNG seed
    syst.seed = 42

    # Setup Real Particles
    for i in range(n_part):
        syst.part.add(id=i, pos=np.random.random(3) * syst.box_l, type=0)

    bulk_potentials(syst, input_file)

    return syst, n_part


def bulk_potentials(system, input_file):
    """

    """
    tables = np.loadtxt(input_file)
    system.force_cap = 0.  # Hard coded

    # skin depth; 0.2 is common choice see pg 556 Frenkel & Smit,
    system.cell_system.skin = 0.2 * tables[0, -1]

    system.non_bonded_inter[0, 0].tabulated.set_params(
        min=tables[0, 0], max=tables[0, -1],
        energy=tables[1, :], force=tables[2, :])

