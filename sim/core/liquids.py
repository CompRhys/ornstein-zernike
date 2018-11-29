from __future__ import print_function
import sys
import os
import re
import espressomd
import numpy as np
import timeit


def initialise_dict(dictionary, rho, n_part):
    """
    Initialises the system based off a dictionary
    """

    # Control the density
    box_l = np.power(n_part / rho, 1.0 / 3.0)
    print('Density={:.2f} \nNumber of Particles={} \nBox Size={:.1f}'.strip().format(rho, n_part, box_l))

    # Box setup
    syst = espressomd.System(box_l=[box_l] * 3)

    # Hardcode PRNG seed
    syst.seed = 42

    # Particle setup
    for i in range(n_part):
        syst.part.add(id=i, pos=np.random.random(3) * syst.box_l, type=0)

    # Interaction
    setup_core_potentials(syst, dictionary)

    return syst


def setup_core_potentials(system, parameters):
    """
    Use routines hard coded into espresso to setup the interaction potentials.
    """

    # skin depth; 0.2 is common choice see pg 556 Frenkel & Smit, 
    syst.cell_system.skin = 0.2 * parameters['cutoff']
    syst.force_cap = 0.  # Hard coded

    if parameters['type'] == 'lj':
        system.non_bonded_inter[0, 0].lennard_jones.set_params(
            epsilon=parameters['energy'], sigma=parameters['sigma'],
            cutoff=parameters['cutoff'], shift='auto')
    elif parameters['type'] == 'soft':
        system.non_bonded_inter[0, 0].soft_sphere.set_params(
            a=parameters['energy'] *
            parameters['sigma']**parameters['exponent'],
            n=parameters['exponent'], cutoff=parameters['cutoff'], shift="auto")
    elif parameters['type'] == 'morse':
        system.non_bonded_inter[0, 0].morse.set_params(
            eps=parameters['energy'], alpha=parameters['alpha'],
            rmin=parameters['min'], cutoff=parameters['cutoff'])
    elif parameters['type'] == 'smooth_step':
        system.non_bonded_inter[0, 0].smooth_step.set_params(
            d=parameters['diameter'], n=parameters['exponent'],
            eps=parameters['energy'], k0=parameters['kappa'],
            sig=parameters['sigma'], cutoff=parameters['cutoff'],
            shift="auto")
    elif parameters['type'] == 'gaussian':
        system.non_bonded_inter[0, 0].gaussian.set_params(
            eps=parameters['energy'], sig=parameters['sigma'],
            cutoff=parameters['cutoff'], shift="auto")
    elif parameters['type'] == 'hertzian':
        system.non_bonded_inter[0, 0].hertzian.set_params(
            eps=parameters['energy'], sig=parameters['energy'])
    elif parameters['type'] == 'hat':
        system.non_bonded_inter[0, 0].hat.set_params(
            F_max=parameters['force'], cutoff=parameters['cutoff'])
    else:
        raise ValueError("no valid interaction type specified")


def initialise_system(input_file, rho, n_part):
    """
    Initialises the system based off an input file
    """

    # Control the density
    box_l = np.power(n_part / rho, 1.0 / 3.0)
    print('Density={:.2f} \nNumber of Particles={} \nBox Size={:.1f}'.strip().format(rho, n_part, box_l))

    # Box setup
    syst = espressomd.System(box_l=[box_l] * 3)

    # Hardcode PRNG seed
    syst.seed = 42

    # Particle setup
    for i in range(n_part):
        syst.part.add(id=i, pos=np.random.random(3) * syst.box_l, type=0)

    # Interaction
    setup_tab_potentials(syst, input_file)

    return syst


def setup_tab_potentials(system, input_file):
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


def disperse_energy(syst, temp, timestep):
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

    energy = syst.analysis.energy()
    min_dist = syst.analysis.min_dist()
    print("Before Minimization: Energy={:.3e}, Min Dist={:.3f}"
          .strip().format(energy['total'], min_dist))
    syst.minimize_energy.init(f_max=10.0, gamma=1.0,
                              max_steps=10000, max_displacement=0.05)
    syst.minimize_energy.minimize()
    energy = syst.analysis.energy()
    min_dist = syst.analysis.min_dist()
    print("After Minimization: Energy={:.3e}, Min Dist={:.3f}"
          .strip().format(energy['total'], min_dist))

    syst.thermostat.recover()
    return min_dist


def disperse_force(syst, temp, timestep, steps, iterations, min_dist, types=[0]):
    """
    This force capping dispersion is computationally cheaper than energy
    minimisation but requires the specification of a minimum allowed distance
    the minimum distance will vary with the interaction potential and determining
    a suitable value is both arbitary and non-trivial.
    """
    print("\nDisperse Particles by force-capping warmup\n")

    syst.thermostat.set_langevin(kT=temp, gamma=1.0)
    n_part = len(syst.part.select())
    syst.time_step = timestep
    syst.force_cap = 5.  # Hard coded
    comb = np.array(np.meshgrid(types, types)).T.reshape(-1, 2)
    act_min_dist = np.zeros(comb.shape[0])

    i = 0
    for j in np.arange(comb.shape[0]):
        act_min_dist[j] = syst.analysis.min_dist(
            p1=[comb[j, 0]], p2=[comb[j, 1]])
        # act_min_dist[j] = syst.analysis.min_dist(p1=syst.part.select(type=comb[j,0]), p2=syst.part.select(type=comb[j,1]))

    for i in np.arange(iterations):
        syst.integrator.run(steps)
        for j in np.arange(comb.shape[0]):
            act_min_dist[j] = syst.analysis.min_dist(
                p1=[comb[j, 0]], p2=[comb[j, 1]])
            # act_min_dist[j] = syst.analysis.min_dist(p1=syst.part.select(type=comb[j,0]), p2=syst.part.select(type=comb[j,1]))
        print("run {} at system time = {:.1f}, max force = {:.1f}, act_min_dist = {}"
              .strip().format(i + 1, syst.time, syst.force_cap, act_min_dist))
        syst.force_cap += 1.

        if all(act_min_dist > min_dist):
            break

    if i == iterations - 1:
        print("\nSystem failed to disperse")

    temp = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
    syst.force_cap = 0.
    print("""Dispersion integration finished at system time = {:.1f}
Temperature at end of integration = {:.3f}""".format(syst.time, temp))
    print('Min_dist at end of integration = ', act_min_dist)


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

    syst.integrator.run(5 * steps)

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

    print("""
Temperature at end of equilibration = {:.3f} +/- {:.3f}
System time at end of equilibration {:.1f}""".format(avg_temp, err_temp, syst.time))


def sample_rdf(syst, timestep, iterations, bins, steps, type_part_a=[0], type_part_b=[0]):
    """
    This function samples the radial distribution function between the two lists of particle
    types a and b. The size of the radius over which we sample is the minimum of the box length
    divided by two or five times the cutoff radius of the potential.
    We also use the inbuild structure factor scheme in order to calculate s(q)
    """
    print("\nSampling RDF\n")

    start = timeit.default_timer()
    n_part = len(syst.part.select())
    r_size = syst.box_l[0] / 2.
    dr = r_size / bins
    r_min = dr / 2.
    r_max = r_size + r_min

    syst.time_step = timestep
    rdf_data = np.zeros((iterations, bins))
    temp = np.zeros(iterations)
    time = np.zeros(iterations)
    for i in range(1, iterations + 1):
        syst.integrator.run(steps)
        r, rdf = syst.analysis.rdf(rdf_type="rdf", type_list_a=type_part_a, type_list_b=type_part_b,
                                   r_min=r_min, r_max=r_max, r_bins=bins)
        rdf_data[i - 1, :] = rdf
        temp[i - 1] = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
        time[i - 1] = syst.time
        if (i % 32) == 0:
            now = timeit.default_timer()
            print("sample run {}/{}, temperature = {:.3f}, system time = {:.1f} (real time = {:.1f})"
                  .strip().format(i, iterations, temp[i - 1], syst.time, now - start))

    return rdf_data, r, temp, time


def sample_sq(syst, timestep, iterations, order, cutoff, steps, type_part=[0]):
    """
    This function samples the structure factor.
    The idea with this section of code was that by directly evaluating s(q) we ensure that it is always 
    positive therefore avoiding issues that later amplify the errors in error propagation scheme.
    Unfortunately the structure factor routine used in ESPResSo doesn't produce an evenly spaced array.
    This leads to significant issues when we try to calculate the inverse 3D fourier transform.
    """
    print("\nSampling S(k)\n")

    start = timeit.default_timer()
    n_part = len(syst.part.select())
    q = syst.analysis.structure_factor(sf_types=type_part, sf_order=order)[0]

    syst.time_step = timestep
    sq_data = np.zeros((iterations, len(q)))
    temp = np.zeros(iterations)
    for i in range(1, iterations + 1):
        syst.integrator.run(steps)
        q, s_q = syst.analysis.structure_factor(
            sf_types=type_part, sf_order=order)
        sq_data[i - 1, :] = s_q
        temp[i - 1] = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
        if (i % 32) == 0:
            now = timeit.default_timer()
            print("sample run {}/{}, temperature = {:.3f}, system time = {:.1f} (real time = {:.1f})"
                  .strip().format(i, iterations, temp[i - 1], syst.time, now - start))
    return sq_data, q, temp


def sample_combo(syst, timestep, iterations, dr, order, steps, type_part=[0]):
    """
    This function samples the radial distribution function between the two lists of particle
    types a and b. The size of the radius over which we sample is the minimum of the box length
    divided by two or five times the cutoff radius of the potential.
    We also use the inbuild structure factor scheme in order to calculate s(q)
    """

    print("\nSampling\n")

    start = timeit.default_timer()
    n_part = len(syst.part.select())
    r_size_max = syst.box_l[0] / 2.
    r = np.arange(dr, r_size_max, dr)
    bins = np.power(2,(np.floor(np.log2(r.shape)))).astype(int)[0]

    r = r[:bins]
    r_min = dr / 2.
    r_max = r[-1] + r_min

    # def find_full_order(order):
    #     values = []
    #     for i in range(order):
    #         for j in range(order):
    #             for k in range(order):
    #                 n = i*i+j*j+k*k
    #                 if n < order**2:
    #                     if n not in values:
    #                         values.append(n)

    #     return len(values)

    syst.time_step = timestep
    rdf_data = np.zeros((iterations, bins))
    sq_data = np.zeros((iterations, order))
    # sq_data = np.zeros((iterations, find_full_order(order)))
    temp = np.zeros(iterations)
    time = np.zeros(iterations)

    time_sq = 0.
    time_rdf = 0.

    for i in range(1, iterations + 1):
        syst.integrator.run(steps)
        start_rdf = timeit.default_timer()
        r, rdf = syst.analysis.rdf(rdf_type="rdf", type_list_a=type_part, type_list_b=type_part,
                                   r_min=r_min, r_max=r_max, r_bins=bins)
        time_rdf += timeit.default_timer() - start_rdf
        start_sq = timeit.default_timer()
        q, s_q = syst.analysis.structure_factor(sf_types=type_part, sf_order=order)
        time_sq += timeit.default_timer() - start_sq
        sq_data[i - 1, :] = s_q
        rdf_data[i - 1, :] = rdf
        temp[i - 1] = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
        time[i - 1] = syst.time 
        if (i % 128) == 0:
            now = timeit.default_timer()
            print("sample run {}/{}, temperature = {:.3f}, system time = {:.1f} (real time = {:.1f})"
                  .strip().format(i, iterations, temp[i - 1], syst.time, now - start))

    print('fraction of time', time_sq / (time_sq + time_rdf))

    return rdf_data, r, sq_data, q, temp, time - time[0]


def sample_phi(syst, radius):
    """
    This Function clears the system and then places particles individually at the sampling points so
    that we end up with a potential function phi(r) that has the same discritisation as our correlation
    functions g(r) and c(r).  
    """
    print("\nSampling Phi\n")
    # Remove Particles
    syst.part.clear()
    bins = len(radius)
    # Place particles and measure interaction energy
    centre = (np.array([0.5, 0.5, 0.5]) * syst.box_l)
    phi = np.zeros_like(radius)

    for i in range(0, bins - 1):
        syst.part.add(pos=centre, id=1, type=0)
        syst.part.add(pos=centre + (radius[i], 0.0, 0.0), id=2, type=0)
        energies = syst.analysis.energy()
        phi[i] = energies['total'] - energies['kinetic']
        syst.part.clear()

        # syst.part[2].remove()

    return phi

