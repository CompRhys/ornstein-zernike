from __future__ import print_function
import espressomd
import numpy as np
from itertools import combinations_with_replacement as cmb

def disperse_energy(syst, temp, timestep, n_test=0):
    """
    This routine moves the particles via gradient descent to a local energy 
    minimium. The parameters f_max, gamma and max_displacement are necessary to
    stop particles shooting off to infinity. The values have been taken from a
    sample script and are used without thought to their justification.
    """

    print("\nDisperse Particles by Minimization of Energy\n")

    # Thermostat
    syst.thermostat.set_langevin(kT=temp, gamma=1.0)
    n_part = len(syst.part.select())
    syst.thermostat.suspend()
    syst.time_step = timestep

    types = range(n_test+1)

    comb = list(cmb(types, 2))
    act_min_dist = np.zeros(len(comb))
    for j in range(len(comb)):
        act_min_dist[j] = syst.analysis.min_dist(
            p1=[comb[j][0]], p2=[comb[j][1]])

    energy = syst.analysis.energy()

    print("Before Minimization: Energy={:.3e}, Min Dist={}"
          .strip().format(energy['total'], act_min_dist))

    # Relax structure
    syst.minimize_energy.init(f_max=10.0, gamma=1.0,
                              max_steps=1000, max_displacement=0.05)
    syst.minimize_energy.minimize()

    for j in range(len(comb)):
        act_min_dist[j] = syst.analysis.min_dist(
            p1=[comb[j][0]], p2=[comb[j][1]])

    energy = syst.analysis.energy()

    print("After Minimization: Energy={:.3e}, Min Dist={}"
          .strip().format(energy['total'], act_min_dist))

    syst.thermostat.recover()
    # return min_dist
    pass


def equilibrate_system(syst, timestep, final_temp, steps, iterations):
    """
    The system is integrated using a small timestep such that the thermostat noise causes
    the system to warm-up. We define the convergence of this equilibration integration
    as the point at which the mean and standard deviation of the last three samples overlaps
    the target temperature.
    """
    print("\nEquilibration\n")

    syst.time_step = timestep
    n_part = len(syst.part.select())

    eq_temp = np.full(3, np.nan)
    avg_temp = 0.
    err_temp = 0.

    # syst.integrator.run(5 * steps)

    temp = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
    print("Initial Temperature = {:.3f}".format(temp))

    i = 0
    while np.abs(avg_temp - final_temp) > err_temp and i < iterations:
        syst.integrator.run(steps)
        kine_energy = syst.analysis.energy()['kinetic']
        eq_temp[i % 3] = kine_energy / (1.5 * n_part)
        avg_temp = np.nanmean(eq_temp)
        # can't have ddof = 1
        err_temp = np.nanstd(eq_temp) / np.sqrt(min(i + 1, 3))
        if np.abs(avg_temp - final_temp) > err_temp:
            print("Equilibration not converged, Temperature = {:.3f} +/- {:.3f}"
                  .format(avg_temp, err_temp))
        np.roll(eq_temp, -1)
        i += 1

    if i == iterations:
        print("\nSystem failed to equilibrate")

    print('\nTemperature at end of equilibration = {:.3f} +/- {:.3f}'
        .format(avg_temp, err_temp, syst.time))
    print('System time at end of equilibration {:.1f}'
        .format(avg_temp, err_temp, syst.time))
