from __future__ import print_function
import sys
import os
import re
import espressomd
import numpy as np
import timeit


def get_bulk(syst, timestep, iterations, dr, order, steps, type_part=[0]):
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
    bins = np.power(2, (np.floor(np.log2(r_size_max / dr)))).astype(int)

    r_min = dr / 2.
    r_max = dr * bins + r_min

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
        q, s_q = syst.analysis.structure_factor(
            sf_types=type_part, sf_order=order)
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


def get_phi(syst, radius, type_a=0, type_b=0):
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
        syst.part.add(pos=centre, id=1, type=type_a)
        syst.part.add(pos=centre + (radius[i], 0.0, 0.0), id=2, type=type_b)
        energies = syst.analysis.energy()
        phi[i] = energies['total'] - energies['kinetic']
        syst.part.clear()

        # syst.part[2].remove()

    return phi


def get_reference(syst, iterations, steps, n_part):
    energy = 0
    energies = []

    for i in range(iterations):
        E_with = syst.analysis.energy()["non_bonded"]
        p = syst.part[n_part - 1]
        pid, pos, v, ptype = (p.id, p.pos, p.v, p.type)
        syst.part[pid].remove()
        E_without = syst.analysis.energy()["non_bonded"]
        syst.part.add(id=pid, pos=pos, v=v, type=ptype)
        energy += np.exp(E_without - E_with)
        energies.append(energy / (i + 1))
        syst.integrator.run(steps)

    print('\n')
    print(energies[-1])

    return energies


def get_external(syst, iterations, diffuse, steps, n_part, dr, bins, limit=2.5):

    # estimates = np.zeros((bins, iterations))
    # estimates = np.zeros(bins)
    energies = np.zeros(bins)
    samples = np.zeros(bins)
    p = syst.part[n_part - 2]
    p2 = syst.part[n_part - 1]

    for i in range(iterations):
        print('run '+str(i)+' out of '+str(iterations))

        pid, pos, v, ptype = (p.id, p.pos, p.v, p.type)
        syst.part[pid].remove()
        # syst.part.add(id=pid, pos=p2.pos + np.random.random(3)
        #               * limit, v=v, type=ptype)
        syst.part.add(id=pid, pos=syst.box_l[0]*np.random.random(3),
                        v=v, type=ptype)
        syst.minimize_energy.minimize()
        syst.integrator.run(steps)

        # energies = np.zeros(bins)
        # samples = np.zeros(bins)
        failed = 0

        for j in range(diffuse):
            r = syst.analysis.min_dist(p1=[1], p2=[2])
            # print(r)

            # print(r)

            index = np.floor(r / dr - 0.5).astype(int)

            if index < 0:
                failed += 1
            else:
                E_with = syst.analysis.energy()["non_bonded"]
                pid, pos, v, ptype = (p.id, p.pos, p.v, p.type)
                syst.part[pid].remove()
                E_without = syst.analysis.energy()["non_bonded"]
                syst.part.add(id=pid, pos=pos, v=v, type=ptype)
                energies[index] += np.exp(E_without - E_with)
                samples[index] += 1

            syst.integrator.run(steps)

        # estimate = np.reshape(energies/samples, (bins,-1))
        # estimates[:,i] = energies/samples

    # todo estimate the error in our estimate using DDA

    estimates = energies / samples

    print('failed: ', failed)

    return estimates
