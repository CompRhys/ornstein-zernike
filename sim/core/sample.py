from __future__ import print_function
import sys
import os
import re
import espressomd
import numpy as np
import timeit

from sklearn.metrics import pairwise_distances as pdist2
import matplotlib.pyplot as plt

from sys import getsizeof

def get_pos_vel(syst, timestep, iterations, steps):
    """
    save the unfolded particle positions and velocity
    """

    print("\nSampling\n")

    start = timeit.default_timer()
    n_part = len(syst.part.select())
    syst.time_step = timestep

    pos = np.zeros((iterations, 3*n_part))
    vel = np.zeros((iterations, 3*n_part))

    temp = np.zeros(iterations)
    time = np.zeros(iterations)

    for i in range(iterations):
        syst.integrator.run(steps)
        pos[i,:] = np.copy(syst.part[:].pos).reshape(-1)
        vel[i,:] = np.copy(syst.part[:].v).reshape(-1)

        temp[i - 1] = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
        time[i - 1] = syst.time
        if (i % 128) == 0:
            now = timeit.default_timer()
            print(('sample run {}/{}, temperature = {:.3f}, '
                   'system time = {:.1f} (real time = {:.1f})').format(
                i, iterations, temp[i - 1], syst.time, now - start))

    return pos, vel, temp, time - time[0]


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
        q, s_q = syst.analysis.structure_factor_fast(
        # q, s_q = syst.analysis.structure_factor(
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
                     r, dr, bins, input_file):
    """
    repeat the cavity sampling for a given number of iterations
    returns:
        cav = (cav) array(iterations,bins)
        mu = (mu) array(iterations)
    """
    print("\nSampling Cavity\n")

    tables = np.loadtxt(input_file)

    start = timeit.default_timer()

    syst.time_step = dt

    cav = np.zeros((iterations, bins))
    mu = np.zeros(iterations)

    cav_repeat = np.ceil(10 * r**2).astype(int)

    for q in range(iterations):
        print('sample run {}/{}'.format(q + 1, iterations))
        mu[q]= get_mu(syst, n_part, mu_repeat, tables)
        cav[q, :] = get_cavity(syst, n_part, cav_repeat, dr, bins, tables)

        syst.integrator.run(steps)
        now = timeit.default_timer()
        print('sample run {}/{} (real time = {:.1f})'.format(
            q + 1, iterations, now - start))

    return cav, mu

def get_cavity(syst, n_part, n_repeat, dr, bins, tables):

    E = syst.analysis.energy()
    curtemp = E['kinetic'] / (1.5 * (n_part))

    # fast calculation of the Cavity correlation
    true_pos = np.vstack(np.copy(syst.part[:].pos_folded))
    # true_pos = np.vstack(syst.part[:].pos)
    # true_pos = np.mod(true_pos, syst.box_l[0])
    cav_fast = np.zeros(bins)

    for k in range(bins):
        for i, true_r in enumerate(true_pos):
            ghost_pos = np.mod(vec_on_sphere(n_repeat[k]) * k * dr + true_r, syst.box_l[0])
            gt_dist = pdist2(np.delete(true_pos, i, axis=0), ghost_pos)
            E_fast = np.sum(np.interp(gt_dist, tables[0, :], tables[1, :]), axis = 0)
            cav_fast[k] += np.mean(np.exp(-E_fast/curtemp), axis = 0)

    cav_fast = cav_fast/n_part

    # # slow calculation of the Cavity correlation
    # cav = np.zeros(bins)
    # cav_count = np.zeros(bins)

    # # for j in range(n_part):
    # for j in range(10):
    #     orig_pos = np.array(syst.part[j].pos)
    #     syst.part[j].type = 1
    #     E_j = syst.analysis.energy()["non_bonded"]
    #     syst.part[j].type = 0
    #     for k in range(bins):
    #         for l in range(n_repeat[k]):
    #             syst.part[j].pos = np.ravel(orig_pos + vec_on_sphere() * k * dr).tolist()
    #             E_refj = syst.analysis.energy()["non_bonded"]
    #             cav[k] += np.exp(-(E_refj - E_j) / curtemp)
    #             cav_count[k] += 1
    #     if ((j+1) % 128) == 0:
    #         print(('checking particle {}/{}').format(j+1, n_part))

    #     syst.part[j].pos = orig_pos

    # cav_slow = cav / cav_count

    # plt.plot(cav_fast)
    # plt.plot(cav_slow)
    # plt.show()

    # exit()

    return cav_fast


def vec_on_sphere(n_repeat=1):
    vec = np.random.randn(3,n_repeat)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), axis=0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag


def get_mu(syst, n_part, n_repeat, tables):
    """ currently mu fast is 10 times larger?"""

    E_ref = syst.analysis.energy()
    curtemp = E_ref['kinetic'] / (1.5 * n_part)

    start = timeit.default_timer()

    # fast calculation of chemical potential
    true_pos = np.vstack(np.copy(syst.part[:].pos_folded))
    # true_pos = np.vstack(syst.part[:].pos)
    # true_pos = np.mod(true_pos, syst.box_l[0])

    ghost_pos = np.random.random((n_repeat, 3)) * syst.box_l
    gt_dist = pdist2(true_pos, ghost_pos)
    E_fast = np.sum(np.interp(gt_dist, tables[0, :], tables[1, :]), axis = 0)
    mu_fast = np.mean(np.exp(-E_fast/curtemp), axis = 0)
    end_fast = timeit.default_timer()

    # # slow calculation of chemical potential
    # E_slow = np.zeros_like(E_fast)
    # syst.part.add(id=n_part + 1, pos=syst.box_l / 2., type=0)
    # for i in range(n_repeat):
    #     syst.part[n_part + 1].pos = ghost_pos[i,:]
    #     E_slow[i] = syst.analysis.energy()["non_bonded"] - E_ref["non_bonded"]
    # syst.part[n_part + 1].remove()
    # mu_slow = np.mean(np.exp(-E_slow/curtemp), axis = 0)
    # end_slow = timeit.default_timer()

    # # print results
    # print("fast {}, time {}".format(mu_fast, start-end_fast))
    # print("slow {}, time {}".format(mu_slow, end_fast-end_slow))

    return mu_fast
