from __future__ import print_function
import sys
import re
import os 
import espressomd
import numpy as np
from core import block, transforms, liquids
from espressomd import visualization
from threading import Thread
import timeit

def main():
    start = timeit.default_timer()

    print("""
    =======================================================
    =                  Program Information                =
    =======================================================
    """)
    print(espressomd.features())

    # Simulation Parameters

    # display_figures         = True
    display_figures         = False

    temperature             = 1.
    timestep                = 0.005

    burn_steps              = 1024
    burn_iterations_max     = 16

    sampling_steps          = 16
    # sampling_iterations     = 16384       # choose 2**n for fp method
    sampling_iterations     = 4096       # choose 2**n for fp method
    r_bins                  = 512       # choose 2**n for fft optimisation
    sq_order                = 15

    # Setup Espresso Environment
    setup_dict = {  'particles':256, 'rho':0.6,'type':'lj', 'energy':1.0, 'sigma':1.0, 'cutoff':2.0}
    system, density = liquids.initialise_dict(temperature, setup_dict)

    # Disperse Particles to energy minimum
    min_dist = liquids.disperse_energy(system, timestep)

    # Integrate the system to warm up to specified temperature
    liquids.equilibrate_system(system, timestep, 
                               temperature, burn_steps,
                               burn_iterations_max)

    def loop():
        while True:
            system.integrator.run(100)
            visualizer.update()

    visualizer = visualization.mayaviLive(system)
    # # visualizer = visualization.openGLLive(system)

    #Start simulation in seperate thread
    t = Thread(target=loop)
    t.daemon = True
    t.start()

    #Start blocking visualizer
    visualizer.start()


if __name__ == "__main__":
    main()

