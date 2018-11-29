import sys
import os
import re
import espressomd
import timeit
import numpy as np
from core import liquids


def main(input_file):
    start = timeit.default_timer()

    # Simulation Parameters

    save = False
    display_figures = True
    # display_figures         = False

    density = 0.7
    n_part = 128

    temperature = 1.
    timestep = 0.005

    burn_steps = 1024
    burn_iterations_max = 16

    sampling_steps = 8
    sampling_iterations = 32    # choose 2**n for fp method
    dr = 0.005

    # Setup Espresso Environment
    system = liquids.initialise_system(input_file, density, n_part)



if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
