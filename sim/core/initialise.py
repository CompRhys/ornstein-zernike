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

    syst.time_step = timestep

    # Thermostat
    syst.thermostat.set_langevin(kT=temp, gamma=1.0, seed=123)
    n_part = len(syst.part.select())
    syst.thermostat.suspend()

    types = range(n_test+1)

    comb = list(cmb(types, 2))
    act_min_dist = np.zeros(len(comb))
    for j in range(len(comb)):
        act_min_dist[j] = syst.analysis.min_dist(
            p1=[comb[j][0]], p2=[comb[j][1]])

    energy = syst.analysis.energy()['non_bonded']

    print("Before Minimization: Energy={:.3e}, Min Dist={}"
          .strip().format(energy, act_min_dist))

    # Relax structure
    syst.integrator.set_steepest_descent(
        f_max=10., gamma=0.1, max_displacement=0.005)
    syst.integrator.run(2000)
    syst.integrator.set_vv()

    # remove force capping
    syst.force_cap = 0

    for j in range(len(comb)):
        act_min_dist[j] = syst.analysis.min_dist(
            p1=[comb[j][0]], p2=[comb[j][1]])

    energy = syst.analysis.energy()['non_bonded']

    print("After Minimization: Energy={:.3e}, Min Dist={}"
          .strip().format(energy, act_min_dist))


    # recover thermostat
    syst.thermostat.recover()
    # return min_dist
    pass


def equilibrate_system(syst, timestep, final_temp, burn, iterations):
    """
    The system is integrated using a small timestep such that the thermostat 
    noise causes the system to warm-up. We define the convergence of this 
    equilibration integration as the point at which the mean and standard 
    deviation of the last three samples overlaps the target temperature.
    """
    print("\nEquilibration\n")

    syst.time_step = timestep
    n_part = len(syst.part.select())

    n_test = 5

    eq_temp = np.full(n_test, np.nan)
    avg_temp = 0.
    err_temp = 0.

    syst.integrator.run(burn)
    temp = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
    print("Initial Temperature after burn-in period = {:.3f}".format(temp))


    i = 0
    while np.abs(avg_temp - final_temp) > err_temp and i < iterations:
        syst.integrator.run(burn)
        kine_energy = syst.analysis.energy()['kinetic']
        eq_temp[i % n_test] = kine_energy / (1.5 * n_part)
        avg_temp = np.nanmean(eq_temp)
        # can't have ddof = 1
        err_temp = np.nanstd(eq_temp) / np.sqrt(min(i + 1, n_test))
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
