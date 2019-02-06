from __future__ import print_function
import espressomd

import os
import numpy as np

from core import initialise, sample
from espressomd import visualization
from threading import Thread

n_part = 2048
density = 0.5

skin = 0.2
dt = 0.005
temperature = 2.5

box_l = np.power(n_part / density, 1.0 / 3.0)
rdf_cut = box_l / 2.

warm_steps = 100
warm_n_time = 2000
min_dist = [0.25, 0.6, 0.6, 0.7]

# integration

burn_steps = 512
burn_iterations_max = 16

sampling_steps = 32
sampling_iterations = 256

r_bins = 512       # choose 2**n for fft optimisation
sq_order = 15


# Interaction parameters (Lennard Jones)
#############################################################

sigma = 1

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
system = espressomd.System(box_l=[box_l] *3)
system.seed = 42

system.time_step = dt
system.cell_system.skin = skin * lj_sig_B


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

# system.force_cap = lj_cap

print("LJ-parameters:")
print(system.non_bonded_inter[0, 0].lennard_jones.get_params())
print(system.non_bonded_inter[1, 1].lennard_jones.get_params())
print(system.non_bonded_inter[0, 1].lennard_jones.get_params())

# Thermostat
system.thermostat.set_langevin(kT=temperature, gamma=2.0)

# Particle setup
#############################################################

volume = box_l * box_l * box_l

for i in range(n_part):
    system.part.add(id=i, pos=np.random.random(3) * system.box_l)
    # Every 2nd particle should be of component 1
    if i % 5 == 1:
        system.part[i].type = 1

density_B = density * \
    (float(len(system.part.select(type=1))) / len(system.part.select()))

#############################################################
#  Warmup Integration                                       #
#############################################################

initialise.disperse_energy(system, temperature, dt)

initialise.equilibrate_system(system, dt,
                              temperature, burn_steps,
                              burn_iterations_max)

sq_order = np.ceil(4 * system.box_l[0] / np.pi).astype(int)
sampling_iterations, sampling_steps, dr = (256, 16, 0.02)

rdf, r, sq, q, temp, t = sample.get_bulk(system, dt,
                                        sampling_iterations, dr,
                                        sq_order, sampling_steps)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, figsize=(10,6))
axes[0].plot(r, np.mean(rdf, axis=1))
axes[1].plot(q, np.mean(sq, axis=0))
plt.show()


# def loop():
#     while True:
#         system.integrator.run(100)
#         visualizer.update()

# # visualizer = visualization.mayaviLive(system)
# # visualizer = visualization.openGLLive(system)
# visualizer = visualization.openGLLive(system, drag_enabled=True, drag_force=100)

# # Start simulation in seperate thread
# t = Thread(target=loop)
# t.daemon = True
# t.start()

# # Start blocking visualizer
# visualizer.start()
