from __future__ import print_function
import sys
import os 
import re
import espressomd
import numpy as np
import timeit

# code graveyard
# sampling_iterations       = 1<<(x-1).bit_length() # this returns next largest power of 2

# Debug Tools
# sys.exit()
# print('ding')

def initialise_dict(temp, dictionary):
    """
    Initialises the system based off a dictionary
    """

    # System parameters
    rho  = dictionary['rho']
    # timestep = dictionary['timestep']

    # Dependent parameters
    # Control the density
    n_part = dictionary['particles']
    box_l = np.power(n_part/rho, 1.0/3.0)
    print("""Density={:.2f}
Number of Particles={}
Box Size={:.1f}""".strip().format(rho,n_part,box_l))
                                    
    # Box setup 
    syst = espressomd.System(box_l=[box_l]*3)
    # Hardcode PRNG seed
    syst.seed = 42
    # verlet lists skin parameter reference Frenkel & Smit, 0.2 is common choice see pg 556.
    syst.cell_system.skin = 0.2 * dictionary['cutoff'] 
    syst.force_cap = 0. # Hard coded
    

    # Thermostat
    syst.thermostat.set_langevin(kT=temp, gamma=1.0) 

    # Particle setup
    for i in range(n_part):
        syst.part.add(id=i, pos=np.random.random(3) * syst.box_l, type=0)

    # Interaction
    setup_core_potentials(syst, dictionary)

    # return file, syst, rho, timestep, cutoff
    return syst, rho

def setup_core_potentials(system, parameters):
    """
    Use routines hard coded into espresso to setup the interaction potentials.
    """

    if parameters['type'] == 'lj':
        system.non_bonded_inter[0, 0].lennard_jones.set_params(
        epsilon=parameters['energy'], sigma=parameters['sigma'],
        cutoff=parameters['cutoff'], shift='auto')
    elif parameters['type'] == 'soft': 
        system.non_bonded_inter[0, 0].soft_sphere.set_params(
        a = parameters['energy']*parameters['sigma']**parameters['exponent'],
        n = parameters['exponent'], cutoff = parameters['cutoff'], shift= "auto")
    elif parameters['type'] == 'morse': 
        system.non_bonded_inter[0, 0].morse.set_params(
        eps= parameters['energy'], alpha = parameters['alpha'], 
        rmin= parameters['min'], cutoff = parameters['cutoff'])
    elif parameters['type'] == 'smooth_step': 
        system.non_bonded_inter[0, 0].smooth_step.set_params(
        d=parameters['diameter'], n=parameters['exponent'], 
        eps= parameters['energy'], k0 = parameters['kappa'], 
        sig= parameters['sigma'], cutoff = parameters['cutoff'],
        shift= "auto")
    elif parameters['type'] == 'gaussian': 
        system.non_bonded_inter[0, 0].gaussian.set_params(
        eps = parameters['energy'], sig = parameters['sigma'], 
        cutoff = parameters['cutoff'], shift= "auto")
    elif parameters['type'] == 'hertzian':
        system.non_bonded_inter[0, 0].hertzian.set_params(
        eps = parameters['energy'], sig = parameters['energy'])
    elif parameters['type'] == 'hat':
        system.non_bonded_inter[0, 0].hat.set_params(
        F_max = parameters['force'], cutoff = parameters['cutoff'])
    else:
        raise ValueError("no valid interaction type specified")


def initialise_system(temp):
    """
    Initialises the system based off an input file
    """
    path = os.path.expanduser('~')+'/Liquids/data'
    if not os.path.exists(path+'/tested/'):
        os.mkdir(path+'/tested')

    tested_path = path+'/tested/'
    input_path  = path+'/input/'

    
    # # Specified Load
    test_number = str(input("Input file number: "))
    file = 'input_'+test_number+'.npy'
    dictionary = np.load(input_path+file).item()

    print('inputfile =',file)

    # System parameters
    rho  = dictionary['rho']
    # timestep = dictionary['timestep']

    # Dependent parameters
    # Control the density
    n_part = dictionary['particles']
    box_l = np.power(n_part/rho, 1.0/3.0)
    print("""Density={:.2f}
Number of Particles={}
Box Size={:.1f}""".strip().format(rho,n_part,box_l))
                                    
    # Box setup 
    syst = espressomd.System(box_l=[box_l]*3)
    # Hardcode PRNG seed
    syst.seed = 42
    # verlet lists skin parameter reference Frenkel & Smit, 0.2 is common choice see pg 556.
    syst.cell_system.skin = 0.2 * dictionary['cutoff'] 
    syst.force_cap = 0. # Hard coded
    

    # Thermostat
    syst.thermostat.set_langevin(kT=temp, gamma=1.0) 

    # Particle setup
    for i in range(n_part):
        syst.part.add(id=i, pos=np.random.random(3) * syst.box_l, type=0)

    # Interaction
    setup_tab_potentials(syst, dictionary, path, file)

    # return file, syst, rho, timestep, cutoff
    return file, syst, rho

def setup_tab_potentials(system, parameters, homepath, test_file):
    """
    In order to be more consistent in out apporach and to facilitate adaptive time-step selection
    based off the hard shell repulsion all the potentials studied will be done via the tabulated 
    method. This is interesting as although we are introducing numerical erros with regard to the
    true analytical expressions we have used for inspiration when it comes to our purpose of
    deriving the best local closure deviations from the analytical inspirations do not matter 
    instead what we are interested in is ensuring that we have consistency across all of our
    measurements.
    """

    test_number = re.findall('\d+', test_file)[0]
    tables = np.loadtxt(homepath+'/tables/input_'+test_number+'.dat')

    system.non_bonded_inter[0, 0].tabulated.set_params(
    min=parameters['min'], max=parameters['cutoff'], # nb max specifies the cutoff value.
    energy=tables[0,:], force=tables[1,:])


def disperse_energy(syst, timestep):
    """
    This routine moves the particles via gradient descent to a local energy 
    minimium. The parameters f_max, gamma and max_displacement are necessary to
    stop particles shooting off to infinity. The values have been taken from a
    sample script and are used without thought to their justification.
    """

    print("\nDisperse Particles by Minimization of Energy\n")

    n_part = len(syst.part.select())
    syst.thermostat.suspend()
    syst.time_step = timestep

    energy = syst.analysis.energy()
    min_dist = syst.analysis.min_dist()
    print("Before Minimization: Energy={:.3e}, Min Dist={:.3f}"
          .strip().format(energy['total'], min_dist))
    syst.minimize_energy.init(f_max = 10.0, gamma = 1.0, 
                max_steps = 10000, max_displacement= 0.05)
    syst.minimize_energy.minimize()
    energy = syst.analysis.energy()
    min_dist = syst.analysis.min_dist()
    print("After Minimization: Energy={:.3e}, Min Dist={:.3f}"
        .strip().format(energy['total'], min_dist))

    syst.thermostat.recover()
    return min_dist



def disperse_force(syst, timestep, steps, iterations, min_dist, types=[0]):
    """
    This force capping dispersion is computationally cheaper than energy
    minimisation but requires the specification of a minimum allowed distance
    the minimum distance will vary with the interaction potential and determining
    a suitable value is both arbitary and non-trivial.
    """
    print("\nDisperse Particles by force-capping warmup\n")

    n_part = len(syst.part.select())
    syst.time_step = timestep
    syst.force_cap = 5. # Hard coded
    comb = np.array(np.meshgrid(types,types)).T.reshape(-1,2)
    act_min_dist = np.zeros(comb.shape[0])

    i = 0
    for j in np.arange(comb.shape[0]):
        act_min_dist[j] = syst.analysis.min_dist(p1=[comb[j,0]], p2=[comb[j,1]])
        # act_min_dist[j] = syst.analysis.min_dist(p1=syst.part.select(type=comb[j,0]), p2=syst.part.select(type=comb[j,1]))

    for i in np.arange(iterations):
        syst.integrator.run(steps)
        for j in np.arange(comb.shape[0]):
            act_min_dist[j] = syst.analysis.min_dist(p1=[comb[j,0]], p2=[comb[j,1]])
            # act_min_dist[j] = syst.analysis.min_dist(p1=syst.part.select(type=comb[j,0]), p2=syst.part.select(type=comb[j,1]))
        print( "run {} at system time = {:.1f}, max force = {:.1f}, act_min_dist = {}"
.strip().format(i+1, syst.time, syst.force_cap, act_min_dist))
        syst.force_cap += 1.

        if all(act_min_dist > min_dist):
            break 

    if i == iterations - 1:
        print("\nSystem failed to disperse")
        
    temp = syst.analysis.energy()['kinetic']/( 1.5 * n_part)
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

    eq_temp  = np.full(3, np.nan)
    avg_temp = 0.
    err_temp = 0.

    syst.integrator.run(5*steps)

    i = 0
    while np.abs(avg_temp - final_temp) > err_temp and i < iterations:
        syst.integrator.run(steps)
        kine_energy = syst.analysis.energy()['kinetic']
        eq_temp[i%3] = kine_energy/( 1.5 * n_part)
        avg_temp = np.nanmean(eq_temp)
        err_temp = np.nanstd(eq_temp)/np.sqrt(min(i+1,3)) # can't have ddof = 1
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
    r_size  = syst.box_l[0]/2.
    dr = r_size / bins
    r_min  = dr / 2.
    r_max  = r_size + r_min

    syst.time_step = timestep
    rdf_data = np.zeros((iterations,bins))
    temp = np.zeros(iterations)
    time = np.zeros(iterations)
    for i in range(1, iterations+1):
        syst.integrator.run(steps)
        r, rdf = syst.analysis.rdf(rdf_type="rdf", type_list_a=type_part_a, type_list_b=type_part_b,
                                   r_min=r_min, r_max=r_max, r_bins=bins)
        rdf_data[i-1,:] = rdf
        temp[i-1] = syst.analysis.energy()['kinetic']/( 1.5 * n_part)
        time[i-1] = syst.time
        if (i % 32) == 0:
            now = timeit.default_timer() 
            print("sample run {}/{}, temperature = {:.3f}, system time = {:.1f} (real time = {:.1f})"
            .strip().format(i, iterations, temp[i-1], syst.time, now-start))
    
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
    sq_data = np.zeros((iterations,len(q)))
    temp = np.zeros(iterations)
    for i in range(1, iterations+1):
        syst.integrator.run(steps)
        q, s_q = syst.analysis.structure_factor(sf_types=type_part, sf_order=order)
        sq_data[i-1,:] = s_q
        temp[i-1] = syst.analysis.energy()['kinetic']/( 1.5 * n_part)
        if (i % 32) == 0:
            now = timeit.default_timer() 
            print("sample run {}/{}, temperature = {:.3f}, system time = {:.1f} (real time = {:.1f})"
            .strip().format(i, iterations, temp[i-1], syst.time, now-start))
    return sq_data, q, temp


def sample_combo(syst, timestep, iterations, bins, order, steps, type_part=[0]):
    """
    This function samples the radial distribution function between the two lists of particle
    types a and b. The size of the radius over which we sample is the minimum of the box length
    divided by two or five times the cutoff radius of the potential.
    We also use the inbuild structure factor scheme in order to calculate s(q)
    """

    print("\nSampling\n")

    start = timeit.default_timer()
    n_part = len(syst.part.select())
    r_size  = syst.box_l[0]/2.
    dr = r_size / bins
    r_min  = dr / 2.
    r_max  = r_size + r_min

    syst.time_step = timestep
    rdf_data = np.zeros((iterations,bins))
    sq_data = np.zeros((iterations,order)) 
    temp = np.zeros(iterations)
    time = np.zeros(iterations)
    time_sq = 0.
    time_rdf = 0.
    for i in range(1, iterations+1):
        syst.integrator.run(steps)
        start_rdf   = timeit.default_timer()      
        r, rdf      = syst.analysis.rdf(rdf_type="rdf", type_list_a=type_part, type_list_b=type_part,
                                   r_min=r_min, r_max=r_max, r_bins=bins)
        time_rdf   += timeit.default_timer() - start_rdf 
        start_sq    = timeit.default_timer()      
        q, s_q      = syst.analysis.structure_factor(sf_types=type_part, sf_order=order)
        time_sq    += timeit.default_timer() - start_sq 
        sq_data[i-1,:]  = s_q
        rdf_data[i-1,:] = rdf
        temp[i-1] = syst.analysis.energy()['kinetic']/( 1.5 * n_part)
        time[i-1] = syst.time
        if (i % 128) == 0:
            now = timeit.default_timer() 
            print("sample run {}/{}, temperature = {:.3f}, system time = {:.1f} (real time = {:.1f})"
            .strip().format(i, iterations, temp[i-1], syst.time, now-start))
    
    print('fraction of time', time_sq / (time_sq+time_rdf))

    return rdf_data, r, sq_data, q, temp, time


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
    centre = (np.array([0.5,0.5,0.5]) * syst.box_l)
    phi = np.zeros_like(radius)

    for i in range(0, bins-1):
        syst.part.add(pos=centre , id=1, type=0)
        syst.part.add(pos=centre+(radius[i],0.0,0.0), id=2, type=0)
        energies = syst.analysis.energy()
        phi[i] = energies['total'] - energies['kinetic']
        syst.part.clear()

        # syst.part[2].remove()

    return phi


def smooth_function(f):
    """
    five point smoothing as detailed on page 204 of Computer Simulation of Liquids.
    """

    g = np.zeros_like(f)

    g[:,0]  = 1./70. * (69*f[:,0]   +  4*f[:,1]  -  6*f[:,2]  + 4*f[:,3]  -   f[:,4] )
    g[:,1]  = 1./35. * ( 2*f[:,0]   + 27*f[:,1]  + 12*f[:,2]  - 8*f[:,3]  + 2*f[:,4] )
    g[:,-2] = 1./35. * ( 2*f[:,-1]  + 27*f[:,-2] + 12*f[:,-4] - 8*f[:,-4] + 2*f[:,-5])
    g[:,-1] = 1./70. * (69*f[:,-1]  +  4*f[:,-2] -  6*f[:,-3] + 4*f[:,-4] -   f[:,-5])

    for i in np.arange(2, f.shape[1]-2):
        g[:,i]  = 1./35. * ( -3*f[:,i-2]   + 12*f[:,i-1]  + 17*f[:,i]  + 12*f[:,i+1]  - 3*f[:,i+2] )

    return g


