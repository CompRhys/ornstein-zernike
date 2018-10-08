from __future__ import print_function
import espressomd

import os 
import numpy as np

from routines import liquids
from espressomd import visualization
from threading import Thread

n_part  = 2048
density = 0.5

skin        = 0.2 
timestep    = 0.005 
temperature = 2.5

box_l       = np.power(n_part/density, 1.0/3.0) 
rdf_cut     = box_l / 2.

warm_steps  = 100
warm_n_time = 2000
min_dist    = [0.25,0.6,0.6,0.7]

# integration

burn_steps              = 1024
burn_iterations_max     = 16

sampling_steps          = 256
sampling_iterations     = 256   

r_bins                  = 512       # choose 2**n for fft optimisation
sq_order                = 15


# Interaction parameters (Lennard Jones)
#############################################################

sigma  = 1

lj_cap = 20

lj_eps_A = 1.0
lj_sig_A = 0.3 * sigma 
lj_cut_A = 2.5 * lj_sig_A

lj_eps_B = 1.0
lj_sig_B = 1.0
lj_cut_B = 2.5 * lj_sig_B
# lj_cut_B = 2**(1./6.) * lj_sig_B

# This is the cutoff of the interaction between species 0 and 1.
# By setting it to 2**(1./6.) *lj_sig, it can be made purely repulsive
lj_eps_AB = 1.0
lj_sig_AB = 1.0
lj_cut_AB = 2.5 * lj_sig_B
# lj_cut_AB =2**(1./6.) * lj_sig


# System setup
#############################################################
system  = espressomd.System(box_l=[1.0, 1.0, 1.0])

system.time_step            = timestep
system.cell_system.skin     = skin * lj_sig_B

system.box_l = [box_l, box_l, box_l]


# Here, lj interactions need to be setup for both components
# as well as for the mixed case of component 0 interacting with
# component 1.

# component 0
system.non_bonded_inter[0, 0].lennard_jones.set_params(
    epsilon=lj_eps_A, sigma=lj_sig_A,
    cutoff=lj_cut_A, shift="auto")

# component 1
system.non_bonded_inter[1, 1].lennard_jones.set_params(
    epsilon=lj_eps_B, sigma=lj_sig_B,
    cutoff=lj_cut_B, shift="auto")

# mixed case
system.non_bonded_inter[0, 1].lennard_jones.set_params(
    epsilon=lj_eps_AB, sigma=lj_sig_AB,
    cutoff=lj_cut_AB, shift="auto")

system.force_cap = lj_cap

print("LJ-parameters:")
print(system.non_bonded_inter[0, 0].lennard_jones.get_params())
print(system.non_bonded_inter[1, 1].lennard_jones.get_params())
print(system.non_bonded_inter[0, 1].lennard_jones.get_params())

# Thermostat
system.thermostat.set_langevin(kT=temperature, gamma=1.0)

# Particle setup
#############################################################

volume = box_l * box_l * box_l

for i in range(n_part):
    system.part.add(id=i, pos=np.random.random(3) * system.box_l)
    # Every 2nd particle should be of component 1 
    if i%5==1: system.part[i].type=1

density_B = density * (float(len(system.part.select(type=1)))/len(system.part.select()))

#############################################################
#  Warmup Integration                                       #
#############################################################

liquids.disperse_force(system, timestep, warm_steps, warm_n_time, min_dist, types=[0,1])

liquids.equilibrate_system(system, timestep, 
                           temperature, burn_steps,
                           burn_iterations_max)

def loop():
    while True:
        system.integrator.run(100)
        visualizer.update()

visualizer = visualization.mayaviLive(system)
# visualizer = visualization.openGLLive(system)

#Start simulation in seperate thread
t = Thread(target=loop)
t.daemon = True
t.start()

#Start blocking visualizer
visualizer.start()
