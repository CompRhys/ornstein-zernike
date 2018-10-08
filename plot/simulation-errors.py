from __future__ import print_function
import sys
import re
import os 
import espressomd
import numpy as np
from routines import flyvbjerg, autocorr, liquids, block
import timeit
import matplotlib.pyplot as plt

start = timeit.default_timer()

print("""
=======================================================
=                  Program Information                =
=======================================================
""")
print(espressomd.features())

# Simulation Parameters

display_figures         = True

temperature             = 1.

burn_steps              = 200
burn_iterations_max     = 20

timestep                = 0.001

sampling_steps              = 1
# sampling_steps              = 100
sampling_iterations         = 128     # choose 2**n for fp method
r_bins                      = 512     # choose 2**n for fft optimisation

# Setup Espresso Environment
test_file, system, density = liquids.initialise_system(temperature)

# Disperse Particles to energy minimum
min_dist = liquids.disperse_energy(system, timestep)

# Integrate the system to warm up to specified temperature
liquids.equilibrate_system(system, timestep, 
                           temperature, burn_steps,
                           burn_iterations_max)

# Sample the RDF for the system
rdf, r, kinetic_temp, t = liquids.sample_rdf(system, timestep, sampling_iterations, 
                                            r_bins, sampling_steps)
err_rdf_fp = flyvbjerg.fp_stderr(rdf)
block_size = block.fp_block_length(rdf)
block_rdf  = block.block_data(rdf, block_size)
avg_rdf    = np.mean(block_rdf, axis=0)
err_rdf_b  = np.sqrt(np.var(block_rdf, axis=0, ddof=1)*block_size/sampling_iterations)

# Uncorrelated method
err_rdf = np.sqrt(np.var(rdf, axis=0, ddof=1)/sampling_iterations)

# Correlation time Method - this tends to be 1.6 times larger than fp method.
err_rdf_ac = autocorr.ac_stderr(rdf)


plt.figure()
plt.plot(r, err_rdf/avg_rdf, label='Standard deviation')
plt.plot(r, err_rdf_ac/avg_rdf, label='Autocorrelation Time')
plt.plot(r, err_rdf_b/avg_rdf, label='Flyvbjerg-Peterson')
plt.ylabel('$\sigma(r) \ / \ g(r)$')
plt.xlabel('$r$')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig(path+'/figures/errors'+str(sampling_steps)+'.png')





