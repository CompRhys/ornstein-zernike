from __future__ import print_function
import sys
import os
import re
import espressomd
import numpy as np
import time
import timeit


def sec_to_str(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m,   60)
    return u"{}:{:02d}:{:02d}".format(int(h),int(m),int(s))

def ProgressBar(it, refresh=10):
    """
    refresh=10: refresh the time estimate at least every 10 sec.
    """

    L = len(it)
    steps = {x:y for x, y in zip(np.linspace(0,L,  min(100,L), endpoint=False, dtype=int), 
                                np.linspace(0,100,min(100,L), endpoint=False, dtype=int))}
                                    
    block = u"\u2588"                                     
    partial_block = ["", u"\u258E",u"\u258C",u"\u258A"] # quarter and half block chars

    start = now = time.time()
    timings = " [0:00:00, -:--:--]"
    for nn,item in enumerate(it):
        if nn in steps:
            done = block*int(steps[nn]/4.0) + partial_block[int(steps[nn]%4)]
            todo = " "*(25-len(done))
            bar = u"{}% |{}{}|".format(steps[nn], done, todo)
            if nn>0:
                now = time.time()
                timings = " [{}, {}]".format(sec_to_str(now-start), sec_to_str((now-start)*(L/float(nn)-1)))
            sys.stdout.write("\r"+bar+timings)
            sys.stdout.flush()
        elif time.time()-now > refresh:
            now = time.time()
            timings = " [{}, {}]".format(sec_to_str(now-start), sec_to_str((now-start)*(L/float(nn)-1)))
            sys.stdout.write("\r"+bar+timings)
            sys.stdout.flush()
        yield item
    
    bar  = u"{:d}% |{}|".format(100, block*25)
    timings = " [{}, 0:00:00]\n".format(sec_to_str(time.time()-start))
    sys.stdout.write("\r"+bar+timings)
    sys.stdout.flush()


def get_bulk(syst, timestep, iterations, steps, type_part=[0]):
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

    r_max = syst.box_l[0] * 0.5
    bins = 1024

    dr = r_max / bins

    r_min = dr / 2.
    r_max = r_max + r_min

    syst.time_step = timestep
    rdf_data = np.zeros((iterations, bins))
    sq_data = np.zeros((iterations, bins))
    temp = np.zeros(iterations)
    time = np.zeros(iterations)

    time_int = 0.
    time_sq = 0.
    time_rdf = 0.

    try:
        for i in ProgressBar(range(1, iterations + 1), 40):
            start_int = timeit.default_timer()
            syst.integrator.run(steps)
            start_rdf = timeit.default_timer()
            time_int += start_rdf - start_int
            r, rdf = syst.analysis.rdf(rdf_type="rdf", type_list_a=type_part,
                                    type_list_b=type_part, r_min=r_min,
                                    r_max=r_max, r_bins=bins)

            start_sq = timeit.default_timer()
            time_rdf += start_sq - start_rdf
            # q, s_q = syst.analysis.structure_factor_uniform(
            q, s_q = syst.analysis.structure_factor_fast(
            # q, s_q = syst.analysis.structure_factor(
                sf_types=type_part, sf_order=bins)
                # sf_types=type_part, sf_order=order)

            time_sq += timeit.default_timer() - start_sq

            sq_data[i - 1, :] = s_q
            rdf_data[i - 1, :] = rdf
            temp[i - 1] = syst.analysis.energy()['kinetic'] / (1.5 * n_part)
            time[i - 1] = syst.time

    except KeyboardInterrupt:
        pass

    tot = time_int + time_rdf + time_sq
    print('\nTotal Time: {}, Split: Integration- {:.2f}, RDF- {:.2f}, SQ- {:.2f}'.format(sec_to_str(tot),time_int/tot, time_rdf/tot, time_sq/tot))

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
