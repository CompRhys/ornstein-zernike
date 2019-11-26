from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms, parse
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
matplotlib.rcParams.update({'font.size': 12})


def main(input_path, input_number, input_bpart, input_cpart,
         input_temp, input_density):

    display_figures = True
    input_path = '/home/reag2/PhD/masters/closure/data/raw/'

    phi = np.loadtxt('{}phi_p{}.dat'.format(input_path, input_number))
    rdf = np.loadtxt('{}rdf_d{}_n{}_t{}_p{}.dat'.format(
        input_path, input_density, input_bpart, input_temp, input_number))
    sq = np.loadtxt('{}sq_d{}_n{}_t{}_p{}.dat'.format(
        input_path, input_density, input_bpart, input_temp, input_number))

    r = rdf[0, :]
    r_bins = len(r)
    rdf = rdf[1:, :]
    phi = phi[1, :]

    q = sq[0, :]
    sq = sq[1:, :]

    # optional 5 point function smoothing
    # rdf = transforms.smooth_function(rdf)
    # sq = transforms.smooth_function(sq)

    # Find block size to remove correlations
    # block_size = block.fp_block_length(rdf)
    block_size = 256
    block_rdf = block.block_data(rdf, block_size)
    block_sq = block.block_data(sq, block_size)


    # RDF
    avg_rdf = np.mean(block_rdf, axis=0)
    err_rdf = np.sqrt(np.var(block_rdf, axis=0, ddof=1) / block_rdf.shape[0])

    # s(q)
    avg_sq = np.mean(block_sq, axis=0)
    err_sq = np.sqrt(np.var(block_sq, axis=0, ddof=1) / block_sq.shape[0])

    # s(q) from fft
    sq_fft, q_fft = transforms.hr_to_sq(r_bins, input_density, block_rdf - 1., r)
    assert(np.all(np.abs(q-q_fft)<1e-12))
    avg_sq_fft = np.mean(sq_fft, axis=0)
    err_sq_fft = np.sqrt(np.var(sq_fft, axis=0, ddof=1) / sq_fft.shape[0])

    # Corrected s(q) using switch

    # cosine switching function

    peak = np.min(np.argmax(block_sq, axis=1))
    # peak = np.max(np.argmax(block_sq, axis=1))

    # before = int(peak/2.)
    before = 0
    after = len(q_fft) - peak 
    switch = (1 + np.cos(np.pi * q[:peak] / q[peak])) * .5
    # switch = (1 + np.cos(np.pi * q[:peak - before] / q[peak - before])) * .5
    switch = np.pad(switch, (before, 0), 'constant', constant_values=(1))
    switch = np.pad(switch, (0, after - before), 'constant', constant_values=(0))
    # switch = np.pad(switch, (0, after), 'constant', constant_values=(0))

    sq_switch = switch * block_sq + (1. - switch) * sq_fft
    avg_sq_switch = np.mean(sq_switch, axis=0)
    err_sq_switch = np.sqrt(np.var(sq_switch, axis=0, ddof=1) / sq_switch.shape[0])

    ## Evaluate c(r)

    # evaluate c(r) from h(r)
    cr_fft, r_fft = transforms.hr_to_cr(r_bins, input_density, block_rdf - 1., r)
    avg_cr_fft = np.mean(cr_fft, axis=0)
    err_cr_fft = np.sqrt(np.var(cr_fft, axis=0, ddof=1) / cr_fft.shape[0])

    # evaluate c(r) from s(q)
    cr_dir, r_dir = transforms.sq_to_cr(r_bins, input_density, block_sq, q)
    avg_cr_dir = np.mean(cr_dir, axis=0)
    err_cr_dir = np.sqrt(np.var(cr_dir, axis=0, ddof=1) / cr_dir.shape[0])

    # evaluate c(r) from corrected s(q)
    cr_swtch, r_swtch = transforms.sq_to_cr(r_bins, input_density, sq_switch, q_fft)
    avg_cr_swtch = np.mean(cr_swtch, axis=0)
    err_cr_swtch = np.sqrt(np.var(cr_swtch, axis=0, ddof=1) / cr_swtch.shape[0])

    # # c(r) by fourier inversion of just convolved term for comparision
    # cr_both = transforms.sq_and_hr_to_cr(r_bins, input_density, block_rdf - 1., r, block_sq, q)
    # avg_cr_both = np.mean(cr_both, axis=0)
    # err_cr_both = np.sqrt(np.var(cr_both, axis=0, ddof=1) / cr_both.shape[0])


    ## Evaluate B(r)

    # bridge = np.log(sw_cav) + avg_cr_swtch - avg_rdf + 1
    # bridge2 = np.log(sw_cav) + avg_cr_fft - avg_rdf + 1

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # plot phi(r)

    axes[0, 0].plot(r, phi)
    axes[0, 0].plot(r, np.zeros(len(r)), '--', color="tab:blue")
    axes[0, 0].set_ylim([-3, 8])
    axes[0, 0].set_xlabel('r/$\sigma$')
    axes[0, 0].set_ylabel('$\phi(r)$')

    # plot g(r)

    axes[0, 1].plot(r, avg_rdf, color="tab:blue")
    axes[0, 1].fill_between(r, avg_rdf + err_rdf, avg_rdf - err_rdf, alpha=0.3)
    # axes[0, 1].plot(r, block_rdf.T, alpha=0.1)
    axes[0, 1].plot(r, np.ones(len(r)), '--', linewidth=1.0)
    axes[0, 1].set_xlabel('r/$\sigma$')
    axes[0, 1].set_ylabel('$g(r)$')

    # plot c(r)

    axes[0, 2].plot(r, avg_cr_swtch, label='$c_{sw}(r)$')
    axes[0, 2].fill_between(r, avg_cr_swtch + err_cr_swtch, avg_cr_swtch - err_cr_swtch, alpha=0.2)

    axes[0, 2].plot(r, avg_cr_fft, label='$c_{fft}(r)$')
    axes[0, 2].fill_between(r, avg_cr_fft + err_cr_fft, avg_cr_fft - err_cr_fft, alpha=0.2)

    # axes[0, 2].plot(r, avg_cr_dir, label='$c_{dir}(r)$')
    # axes[0, 2].fill_between(r, avg_cr_dir + err_cr_dir, avg_cr_dir - err_cr_dir, alpha=0.2)

    axes[0, 2].plot(r, np.zeros(len(r)), '--', color="tab:blue")
    axes[0, 2].set_xlabel('r/$\sigma$')
    axes[0, 2].set_ylabel('$c(r)$')
    axes[0, 2].legend()

    # Plot s(q)

    axes[1, 0].plot(q, avg_sq, linewidth=1, marker='x', label='$S_{dir}(q)$')
    axes[1, 0].fill_between(q, avg_sq + err_sq, avg_sq - err_sq, alpha=0.2)

    axes[1, 0].plot(q_fft, avg_sq_fft, linewidth=1, marker='x', label='$S_{fft}(q)$')
    axes[1, 0].fill_between(q_fft, avg_sq_fft + err_sq_fft, avg_sq_fft - err_sq_fft, alpha=0.2)

    axes[1, 0].plot(q_fft, avg_sq_switch, color='g', marker='x', linewidth=1, label='$S_{sw}(q)$')
    axes[1, 0].fill_between(q_fft, avg_sq_switch + err_sq_switch, avg_sq_switch - err_sq_switch, alpha=0.2)

    axes[1, 0].plot(q_fft, switch, color='r',linewidth=1, marker='x', label='W(q)')
    axes[1, 0].set_xlim([0, 12.5])
    # axes[1, 0].set_ylim([-.5, 2.0])
    axes[1, 0].set_xlabel('$q$')
    axes[1, 0].set_ylabel('$S(q), W(q)$')
    axes[1, 0].legend()

    # # plot y(r)

    # axes[1, 1].plot(r, np.arcsinh(bulk_cav), label='$y_{dir}(r)$')
    # axes[1, 1].plot(r2, np.arcsinh(dir_cav), label='$y_{cav}(r)$')
    # axes[1, 1].fill_between(r2, np.arcsinh(dir_cav + err_cav),
    #                         np.arcsinh(dir_cav - err_cav), alpha=0.1)
    # axes[1, 1].plot(r2, np.arcsinh(dirm_cav), label='$y_{mean}(r)$')
    # # axes[1, 1].plot(llanoy[:, 0], np.arcsinh(
    # #     np.exp(llanoy[:, 3])), label='$y_{llano}(r)$')
    # axes[1, 1].plot(r, np.arcsinh(sw_cav), label='$y_{sw}(r)$')
    # axes[1, 1].plot(r, rswitch, label='W(r)')
    # # axes[1, 1].set_ylim([0,5])
    # axes[1, 1].set_xlabel('r/$\sigma$')
    # axes[1, 1].set_ylabel('$y(r)$')
    # axes[1, 1].legend()

    # # plot b(r)

    # axes[1, 2].plot(r, bridge, label='$y_{sw}(r)$')
    # # axes[1, 2].plot(llanob[:, 0], llanob[:, 2], label='$y_{llano}(r)$')
    # axes[1, 2].plot(r, bridge2, label='$y_{fft}(r)$')
    # # axes[1, 2].set_xlim([0, 2])
    # axes[1, 2].set_xlabel('r/$\sigma$')
    # axes[1, 2].set_ylabel('$B(r)$')
    # axes[1, 2].legend(loc=4)

    fig.tight_layout()

    f, axarr = plt.subplots(2, figsize=(6,6), sharex=True, 
        gridspec_kw={'height_ratios':[3, 1]})

    axarr[0].plot(q_fft, avg_sq_fft, linewidth=1,
                  marker='x', label='$S_{fft}(q)$')
    axarr[0].plot(q, avg_sq, linewidth=1,
                  marker='x', label='$S_{dir}(q)$')
    # axarr[0].plot(q_fft, avg_sq_switch,
    #               color='g', marker='x', linewidth=1,
    #               label='$S_{hybrid}(q)$')
    # axarr[0].plot(q_fft, qswitch, color='r',
    #                 linewidth=1, marker='x', label='W(q)')
    axarr[0].set_ylabel('$S(q), W(q)$')
    axarr[0].legend()

    axarr[1].plot(q, (- avg_sq + avg_sq_fft), 
        linewidth=1, marker='x', label='$\Delta S(q)$')
    # axarr[1].plot(q_sam, (- avg_sq + avg_sq_fft[:len(q_sam)]) , 
    #     linewidth=1, marker='x', label='$\Delta S(q)$')
    axarr[1].plot((0,13), (0,0), 'k-.', linewidth=0.5)
    axarr[1].set_xlabel('$q$')
    axarr[1].set_ylabel('$\Delta S(q)$')
    # axarr[1].set_xlim([0, 12.5])
    axarr[1].set_ylim([-0.1, 0.3])
    # axarr[1].legend()
    f.tight_layout()

    plt.show()


if __name__ == "__main__":
    opt = parse.parse_input()
    input_path = opt.output
    input_number = re.findall('\d+', opt.table)[-1]
    input_bpart = opt.bulk_part
    input_cpart = opt.cav_part
    input_density = opt.rho
    input_temp = opt.temp

    print(opt)

    main(input_path, input_number, input_bpart, input_cpart,
         input_temp, input_density)
