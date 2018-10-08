from __future__ import print_function
import sys
import re
import os 
import espressomd
import numpy as np
from core import block, transforms, liquids
import timeit

def main():
    start = timeit.default_timer()

    print("""
    =======================================================
    =                  Program Information                =
    =======================================================
    """)
    print(espressomd.features())

    # Simulation Parameters

    display_figures         = True
    # display_figures         = False

    temperature             = 1.
    timestep                = 0.005

    burn_steps              = 1024
    burn_iterations_max     = 16

    sampling_steps          = 8
    # sampling_iterations     = 16384       # choose 2**n for fp method
    sampling_iterations     = 4096      # choose 2**n for fp method
    # sampling_iterations     = 512     # choose 2**n for fp method
    # sampling_iterations     = 128     # choose 2**n for fp method
    r_bins                  = 512      # choose 2**n for fft optimisation
    sq_order                = 30

    # Setup Espresso Environment
    setup_dict = {  'particles':256, 'rho':0.6, 
                    'type':'lj', 'energy':1.0, 'sigma':1.0, 'cutoff':2.0}
    system, density = liquids.initialise_dict(temperature, setup_dict)

    # Disperse Particles to energy minimum
    min_dist = liquids.disperse_energy(system, timestep)

    # Integrate the system to warm up to specified temperature
    liquids.equilibrate_system(system, timestep, 
                               temperature, burn_steps,
                               burn_iterations_max)

    # Sample the RDF for the system
    rdf, r, sq, q, kinetic_temp, t = liquids.sample_combo(system, timestep, sampling_iterations, 
                                                            r_bins, sq_order, sampling_steps)

    rdf = liquids.smooth_function(rdf)
    sq = liquids.smooth_function(sq)

    # Find block size to remove correlations
    block_size          = block.fp_block_length(rdf)
    # print(block_size)

    # RDF
    block_rdf           = block.block_data(rdf, block_size)
    avg_rdf             = np.mean(block_rdf, axis=0)
    err_rdf             = np.sqrt(np.var(block_rdf, axis=0, ddof=1)/block_rdf.shape[0])

    # Structure Factor via Fourier Transform
    sq_fft, q_fft       = transforms.hr_to_sq(r_bins, density, block_rdf - 1., r)
    avg_sq_fft          = np.mean(sq_fft, axis=0)
    avg_sq              = np.mean(sq, axis=0)

    # Correction of low q limit via direct measurement (Frankenstien's)
    block_sq            = block.block_data(sq, block_size)
    switch              = 1./(1.+np.exp(0.25*system.box_l[0]*(q-(q[-1]-6*q[0])))) # 0.25 prefactor chosen manually should be ~1/(2*pi) from scaling
    sq_pad              = np.pad(avg_sq, (0,len(q_fft)-len(q)), 'constant')
    switch_extend       = np.pad(switch, (0,len(q_fft)-len(q)), 'constant')
    sq_extend           = np.pad(block_sq, ((0,0),(0,len(q_fft)-len(q))), 'constant')
    sq_switch           = switch_extend * sq_extend + (1.-switch_extend)*sq_fft
    avg_sq_switch       = np.mean(sq_switch, axis=0)
    err_sq_switch       = np.sqrt(np.var(sq_switch, axis=0, ddof=1)/sq_switch.shape[0])

    # evaluate c(r) from corrected structure factor
    cr_swtch, r_swtch   = transforms.sq_to_cr(r_bins, density, sq_switch, q_fft)
    avg_cr_swtch        = np.mean(cr_swtch, axis=0)
    err_cr_swtch        = np.sqrt(np.var(cr_swtch, axis=0, ddof=1)/cr_swtch.shape[0])

    # c(r) by fourier inversion for comparision
    cr_fft              = transforms.hr_to_cr(r_bins, density, block_rdf-1, r)
    avg_cr_fft          = np.mean(cr_fft, axis=0)
    err_cr_fft          = np.sqrt(np.var(cr_fft, axis=0, ddof=1)/cr_fft.shape[0])

    # Extract the interaction potential used by the model
    phi = liquids.sample_phi(system, r)

    # Make sure paths exist
    path = os.path.expanduser('~')
    if not os.path.exists(path+'/closure/test'):
        os.mkdir(path+'/closure/test')
    if not os.path.exists(path+'/closure/test/output'):
        os.mkdir(path+'/closure/test/output')
    if not os.path.exists(path+'/closure/test/temperature'):
        os.mkdir(path+'/closure/test/temperature')

    # Save output
    output_path = path+'/closure/test/output/'
    output = np.column_stack((r, phi, avg_rdf, err_rdf, avg_cr_swtch, err_cr_swtch,    
    #                         0   1      2        3          4             5 
            q_fft, avg_cr_fft, err_cr_fft, avg_sq_switch, err_sq_switch, sq_pad, avg_sq_fft, switch_extend))        
    #         6         7           8           9               10         11        12            13
    np.savetxt(output_path+'/test_output.dat', output)

    # Save temperature
    temp_path   = path+'/closure/test/temperature/'
    temp_series = np.column_stack((t,kinetic_temp))
    np.savetxt(temp_path+'/test_temp.dat', temp_series)

    stop = timeit.default_timer()
    print("Total runtime = {}".strip().format(stop-start))

    if display_figures:
        import matplotlib.pyplot as plt

        # Plot g(r) and c(r)

        fig, axes = plt.subplots(2,figsize=(10,10))
        axes[0].errorbar(r, avg_rdf, err_rdf, 
            linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
        axes[0].plot(r, np.ones(len(r)), 'b--', linewidth=0.5)
        axes[1].set_xlabel('r/$\sigma$')
        axes[1].set_ylabel('$g(r)$')
        axes[1].errorbar(r, avg_cr_fft, err_cr_fft, 
            linewidth=1, errorevery=20, ecolor='r', elinewidth=1)    
        axes[1].errorbar(r, avg_cr_swtch, err_cr_swtch, color='g',
            linewidth=1, errorevery=20, ecolor='g', elinewidth=1)
        axes[1].plot(r, np.zeros(len(r)), 'b--', linewidth=0.5)
        axes[1].set_xlabel('r/$\sigma$')
        axes[1].set_ylabel('$c(r)$')

        # Plot s(q)

        plt.figure(figsize=(10,10))
        plt.plot(q, avg_sq, linewidth=1, marker='x', label='$S_{dir}(q)$')
        plt.plot(q_fft, avg_sq_fft, linewidth=1, marker='x', label='$S_{fft}(q)$')
        plt.errorbar(q_fft, avg_sq_switch, err_sq_switch, color='g', marker='x',
            linewidth=1, errorevery=20, ecolor='g', elinewidth=1, label='$S_{avg}(q)$')
        plt.plot(q_fft, switch_extend, color='r', linewidth=1, marker='x', label='W(q)')
        plt.xlim([0,12.5])
        plt.ylim([0,2.0])
        plt.xlabel('$q$')
        plt.ylabel('$S(q), W(q)$')
        plt.legend()
        plt.tight_layout()

        # plt.figure()
        # plt.plot(r, phi, linewidth=1, marker='x')
        # plt.ylim([-5,10])

        plt.show()

if __name__ == "__main__":
    main()


