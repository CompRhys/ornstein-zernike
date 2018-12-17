from __future__ import print_function
import sys
import os
import re
import espressomd
import numpy as np
import timeit


def get_bulk(syst, timestep, iterations, dr, order, steps, type_part=[0]):
    """
    This function samples the radial distribution function between the two 
    lists of particle types a and b. The size of the radius over which we 
    sample is the minimum of the box length divided by two or five times 
    the cutoff radius of the potential. We also use the inbuild structure 
    factor scheme in order to calculate s(q)
    """

    print("\nSampling\n")

    start = timeit.default_timer()
    n_part = len(syst.part.select())

    r_size_max = syst.box_l[0] * 0.5
    bins = np.power(2, (np.floor(np.log2(r_size_max / dr)))).astype(int)

    r_min = dr / 2.
    r_max = dr * bins + r_min

    syst.time_step = timestep
    rdf_data = np.zeros((iterations, bins))
    sq_data = np.zeros((iterations, order))
    temp = np.zeros(iterations)
    time = np.zeros(iterations)

    time_sq = 0.
    time_rdf = 0.

    for i in range(1, iterations + 1):
        syst.integrator.run(steps)
        start_rdf = timeit.default_timer()
        r, rdf = syst.analysis.rdf(rdf_type="rdf", type_list_a=type_part,
                                   type_list_b=type_part, r_min=r_min,
                                   r_max=r_max, r_bins=bins)
        time_rdf += timeit.default_timer() - start_rdf
        start_sq = timeit.default_timer()
        q, s_q = syst.analysis.structure_factor(
            sf_types=type_part, sf_order=order)
        time_sq += timeit.default_timer() - start_sq
        sq_data[i - 1, :] = s_q
        rdf_data[i - 1, :] = rdf
        temp[i - 1] = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
        time[i - 1] = syst.time
        if (i % 128) == 0:
            now = timeit.default_timer()
            print(('sample run {}/{}, temperature = {:.3f}, '
                   'system time = {:.1f} (real time = {:.1f})').format(
                i, iterations, temp[i - 1], syst.time, now - start))

    print('fraction of time', time_sq / (time_sq + time_rdf))

    return rdf_data, r, sq_data, q, temp, time - time[0]


def get_phi(syst, radius, type_a=0, type_b=0):
    """
    This Function clears the system and then places particles individually 
    at the sampling points so that we end up with a potential function phi(r) 
    that has the same discritisation as our correlation functions g(r) and c(r).  
    """
    print("\nSampling Phi\n")
    # Remove Particles
    syst.part.clear()
    bins = len(radius)
    # Place particles and measure interaction energy
    centre = (np.array([0.5, 0.5, 0.5]) * syst.box_l)
    phi = np.zeros_like(radius)

    for i in range(0, bins - 1):
        syst.part.add(pos=centre, id=1, type=type_a)
        syst.part.add(pos=centre + (radius[i], 0.0, 0.0), id=2, type=type_b)
        energies = syst.analysis.energy()
        # phi[i] = energies['total'] - energies['kinetic']
        phi[i] = energies['non_bonded']
        syst.part.clear()

        # syst.part[2].remove()

    return phi


def sample_henderson(syst, dt, iterations, steps, n_part, mu_repeat,
                     r, dr, bins):
    """
    repeat the cavity sampling for a given number of iterations
    returns:
        cav = (cav) array(iterations,bins)
        mu = (mu) array(iterations)
    """
    print("\nSampling Cavity\n")

    start = timeit.default_timer()

    syst.time_step = dt
    cav = np.zeros((iterations, bins))
    mu = np.zeros(iterations)

    cav_repeat = np.ceil(10 * r**2).astype(int)

    for q in range(iterations):
        print('sample run {}/{}'.format(q + 1, iterations))
        mu[q] = get_mu(syst, n_part, mu_repeat)
        cav[q, :] = get_cavity(syst, n_part, cav_repeat, dr, bins)

        syst.integrator.run(steps)
        now = timeit.default_timer()
        print('sample run {}/{} (real time = {:.1f})'.format(
            q + 1, iterations, now - start))

    return cav, mu


def sample_cavity(syst, dt, iterations, steps, n_part, r, dr, bins):
    """
    repeat the cavity sampling for a given number of iterations

    returns:
        cav = (cav) array(iterations,bins)
    """

    start = timeit.default_timer()

    syst.time_step = dt
    cav = np.zeros((iterations, bins))

    cav_repeat = np.ceil(10 * r**2).astype(int)

    for q in range(iterations):
        cav[q, :] = get_cavity(syst, n_part, cav_repeat, dr, bins)
        syst.integrator.run(steps)
        now = timeit.default_timer()
        print('sample run {}/{} (real time = {:.1f})'.format(
            q + 1, iterations, now - start))

    return cav


def get_cavity(syst, n_part, n_repeat, dr, bins):

    E = syst.analysis.energy()
    curtemp = E['kinetic'] / (1.5 * (n_part))

    cav = np.zeros(bins)
    cav_count = np.zeros(bins)

    for j in range(n_part):
        orig_pos = syst.part[j].pos
        syst.part[j].type = 1
        E_j = syst.analysis.energy()["non_bonded"]
        syst.part[j].type = 0
        for k in range(bins):
            for l in range(n_repeat[k]):
                syst.part[j].pos = orig_pos + vec_on_sphere() * k * dr
                E_refj = syst.analysis.energy()["non_bonded"]
                cav[k] += np.exp(-(E_refj - E_j) / curtemp)
                cav_count[k] += 1
        if ((j+1) % 128) == 0:
            print(('checking particle {}/{}').format(j+1, n_part))

        syst.part[j].pos = orig_pos

    avg_cav = cav / cav_count

    return avg_cav


def vec_on_sphere():
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def sample_mu(syst, dt, iterations, steps, n_repeat, n_part):
    start = timeit.default_timer()

    syst.time_step = dt
    mu = np.zeros(iterations)

    for q in range(iterations):
        mu[q] = get_mu(syst, n_part, n_repeat)
        syst.integrator.run(steps)
        now = timeit.default_timer()
        print('sample run {}/{} (real time = {:.1f})'.format(
            q + 1, iterations, now - start))

    return mu


def get_mu(syst, n_part, n_repeat):

    E_ref = syst.analysis.energy()["non_bonded"]
    curtemp = syst.analysis.energy()['kinetic'] / (1.5 * n_part)

    mu = 0

    # calculate the chemical potential of the resevoir
    syst.part.add(id=n_part + 1, pos=syst.box_l / 2, type=0)
    for i in range(n_repeat):
        syst.part[n_part + 1].pos = np.random.random(3) * syst.box_l
        E_mu = syst.analysis.energy()["non_bonded"]
        mu += np.exp(-(E_mu - E_ref) / curtemp)
    syst.part[n_part + 1].remove()

    return mu / n_repeat
