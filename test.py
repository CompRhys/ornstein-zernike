from __future__ import print_function
import espressomd
from espressomd.observables import ParticlePositions
from espressomd.accumulators import Correlator
import numpy as np
import matplotlib.pyplot as plt

def main():
    required_features = ["LENNARD_JONES"]
    espressomd.assert_features(required_features)


    # ### System setup

    # Importing other relevant python modules
    # System parameters
    n_part = 2048
    density = 0.6

    box_l=np.power(n_part/density, 1.0/3.0)*np.ones(3)

    # The next step would be to create an instance of the System class and to seed espresso. This instance is used as a handle to the simulation system. At any time, only one instance of the System class can exist.

    system = espressomd.System(box_l=box_l)
    system.seed = 42

    # It can be used to manipulate the crucial system parameters like the time step and the size of the simulation box (<tt>time_step</tt>, and <tt>box_l</tt>).

    skin = 0.4
    time_step = 0.01
    eq_tstep = 0.001
    temperature = 1.0

    system.time_step = time_step

    system.thermostat.turn_off()

    system.thermostat.set_langevin(kT=temperature, gamma=1.0)

    # Add particles to the simulation box at random positions
    for i in range(n_part):
        system.part.add(type=0, pos=np.random.random(3) * system.box_l)

    # ### Setting up non-bonded interactions

    # Non-bonded interactions act between all particles of a given combination of particle types. In this tutorial, we use the Lennard-Jones non-bonded interaction. The interaction of two particles of type 0 can be setup as follows:

    lj_eps = 1.0
    lj_sig = 1.0
    lj_cut = 2.5*lj_sig
    lj_cap = 0.5
    system.non_bonded_inter[0, 0].lennard_jones.set_params(epsilon=lj_eps, sigma=lj_sig,
    cutoff=lj_cut, shift='auto')
    system.force_cap=lj_cap

    # ### Warmup

    warm_steps  = 100
    warm_n_time = 200
    min_dist    = 0.87

    i = 0
    act_min_dist = system.analysis.min_dist()
    while i < warm_n_time and act_min_dist < min_dist :
        system.integrator.run(warm_steps)
        act_min_dist = system.analysis.min_dist()
        i+=1
        lj_cap += 1.0
        system.force_cap=lj_cap

    # ### Integrating equations of motion and taking measurements

    # Once warmup is done, the force capping is switched off by setting it to zero.
    system.force_cap=0

    # Integration parameters
    sampling_interval       = 10
    sampling_iterations     = 320


    # Pass the ids of the particles to be tracked to the observable.
    part_pos=ParticlePositions(ids=range(n_part))

    # Initialize MSD correlator
    msd_corr=Correlator(obs1=part_pos,
                    tau_lin=10,delta_N=10,
                    tau_max=10000*time_step,
                    corr_operation="square_distance_componentwise")

    # Calculate results automatically during the integration
    system.auto_update_accumulators.add(msd_corr)

    # Set parameters for the radial distribution function
    r_bins = 50
    r_min  = 0.0
    r_max  = system.box_l[0]/2.0

    avg_rdf=np.zeros((r_bins,))
    order = 20
    avg_sq=np.zeros((order,))

    # Take measurements
    time = np.zeros(sampling_iterations)
    instantaneous_temperature = np.zeros(sampling_iterations)
    etotal = np.zeros(sampling_iterations)

    for i in range(1, sampling_iterations + 1):
        system.integrator.run(sampling_interval)
        # Measure radial distribution function
        r, rdf = system.analysis.rdf(rdf_type="rdf", type_list_a=[0], type_list_b=[0], r_min=r_min, r_max=r_max, r_bins=r_bins)
        avg_rdf += rdf/sampling_iterations
        q, sq = system.analysis.structure_factor(sf_types=[0], sf_order=order)
        avg_sq += sq/sampling_iterations

        
        # Measure energies
        energies = system.analysis.energy()
        kinetic_temperature = energies['kinetic']/( 1.5 * n_part)
        etotal[i-1] = energies['total']
        time[i-1] = system.time
        instantaneous_temperature[i-1] = kinetic_temperature

        if (i % 32) == 0:
            print("sample run {}/{}".strip().format(i, sampling_iterations)) 
        
    # Finalize the correlator and obtain the results
    msd_corr.finalize()
    msd=msd_corr.result()

    plt.ion()
    fig1 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    fig1.set_tight_layout(False)
    plt.plot(r, avg_rdf,'-', color="#A60628", linewidth=2, alpha=1)
    plt.xlabel('$r$',fontsize=20)
    plt.ylabel('$g(r)$',fontsize=20)
    plt.show()

    plt.ion()
    fig2 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    fig2.set_tight_layout(False)
    plt.plot(q, avg_sq,'-', color="#A60628", linewidth=2, alpha=1)
    plt.xlabel('$r$',fontsize=20)
    plt.ylabel('$g(r)$',fontsize=20)
    plt.show()

    # fig3 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    # fig3.set_tight_layout(False)
    # plt.plot(msd[:,0], msd[:,2]+msd[:,3]+msd[:,4],'o-', color="#348ABD", linewidth=2, alpha=1)
    # plt.xlabel('Time',fontsize=20)
    # plt.ylabel('Mean squared displacement',fontsize=20)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # fig4 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    # fig4.set_tight_layout(False)
    # plt.plot(time, instantaneous_temperature,'-', color="red", linewidth=2, alpha=0.5, label='Instantaneous Temperature')
    # plt.plot([min(time),max(time)], [temperature]*2,'-', color="#348ABD", linewidth=2, alpha=1, label='Set Temperature')
    # plt.xlabel('Time',fontsize=20)
    # plt.ylabel('Temperature',fontsize=20)
    # plt.legend(fontsize=16,loc=0)
    # plt.show()

    # ### Simple Error Estimation on Time Series Data

    # calculate the standard error of the mean of the total energy
    standard_error_total_energy=np.sqrt(etotal.var())/np.sqrt(sampling_iterations)
    print(standard_error_total_energy)

    input("Press Enter to continue...")


if __name__ == "__main__":
    main()




