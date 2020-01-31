from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms, parse
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
matplotlib.rcParams.update({'font.size': 12})


def main(input_path, pot_type, pot_number, box_size, temp, input_density):

    n_part = int(input_density * (box_size**3.))
    density = n_part / (box_size**3.)

    phi = np.loadtxt('{}phi_{}_{}.dat'.format(input_path, pot_type, pot_number))
    rdf = np.loadtxt('{}rdf_{}_{}_p{}_n{}_t{}.dat'.format(
        input_path, pot_type, pot_number, input_density, n_part, temp))
    sq = np.loadtxt('{}sq_{}_{}_p{}_n{}_t{}.dat'.format(
        input_path, pot_type, pot_number, input_density, n_part, temp))

    r = rdf[0, :]
    r_bins = len(r)
    tcf = rdf[1:, :] - 1.
    phi = phi[1, :]

    q = sq[0, :]
    sq = sq[1:, :]

    # optional 5 point function smoothing
    tcf = transforms.smooth_function(tcf)
    sq = transforms.smooth_function(sq)

    # Find block size to remove correlations
    block_size_tcf = block.fp_block_length(tcf)
    block_size_sq = block.fp_block_length(sq)
    block_size = np.max((block_size_tcf, block_size_sq))
    print("number of observations is {}, \nblock size is {}. \npercent {}%.".format(rdf.shape[0]-1, block_size, block_size/rdf.shape[0]*100))

    # block_size = 256
    block_tcf = block.block_data(tcf, block_size)
    block_sq = block.block_data(sq, block_size)

    # # optional 5 point function smoothing
    # block_tcf = transforms.smooth_function(block_tcf)
    # block_sq = transforms.smooth_function(block_sq)

    # TCF
    avg_tcf = np.mean(block_tcf, axis=0)
    err_tcf = np.sqrt(np.var(block_tcf, axis=0, ddof=1) / block_tcf.shape[0])

    grad_tcf = np.gradient(block_tcf, r, axis=1)
    avg_grad_tcf = np.mean(grad_tcf ,axis=0)
    err_grad_tcf = np.sqrt(np.var(grad_tcf, axis=0, ddof=1) / block_tcf.shape[0])

    # s(q)
    avg_sq = np.mean(block_sq, axis=0)
    err_sq = np.sqrt(np.var(block_sq, axis=0, ddof=1) / block_sq.shape[0])

    # s(q) from fft
    sq_fft, q_fft = transforms.hr_to_sq(r_bins, density, block_tcf, r)
    assert(np.all(np.abs(q-q_fft)<1e-10))
    avg_sq_fft = np.mean(sq_fft, axis=0)
    err_sq_fft = np.sqrt(np.var(sq_fft, axis=0, ddof=1) / sq_fft.shape[0])

    # Switching function w(q)

    # print(np.argmax(block_sq > 0.75*np.max(block_sq), axis=1))
    peak = np.median(np.argmax(block_sq > 0.75*np.max(block_sq), axis=1)).astype(int)

    after = len(q_fft) - peak 
    switch = (1 + np.cbrt(np.cos(np.pi * q[:peak] / q[peak]))) / 2.
    switch = np.pad(switch, (0, after), 'constant', constant_values=(0))


    # Corrected s(q) using switch

    sq_switch = switch * block_sq + (1. - switch) * sq_fft
    avg_sq_switch = np.mean(sq_switch, axis=0)
    err_sq_switch = np.sqrt(np.var(sq_switch, axis=0, ddof=1) / sq_switch.shape[0])

    ## Evaluate c(r)

    # evaluate c(r) from h(r)
    dcf_fft, r_fft = transforms.hr_to_cr(r_bins, density, block_tcf, r)
    avg_dcf_fft = np.mean(dcf_fft, axis=0)
    err_dcf_fft = np.sqrt(np.var(dcf_fft, axis=0, ddof=1) / dcf_fft.shape[0])

    # evaluate c(r) from s(q)
    dcf_dir, r_dir = transforms.sq_to_cr(r_bins, density, block_sq, q)
    avg_dcf_dir = np.mean(dcf_dir, axis=0)
    err_dcf_dir = np.sqrt(np.var(dcf_dir, axis=0, ddof=1) / dcf_dir.shape[0])

    # evaluate c(r) from corrected s(q)
    dcf_swtch, r_swtch = transforms.sq_to_cr(r_bins, density, sq_switch, q_fft)
    avg_dcf_swtch = np.mean(dcf_swtch, axis=0)
    err_dcf_swtch = np.sqrt(np.var(dcf_swtch, axis=0, ddof=1) / dcf_swtch.shape[0])

    grad_dcf_swtch = np.gradient(dcf_swtch, r_swtch, axis=1)
    avg_grad_dcf_swtch = np.mean(grad_dcf_swtch ,axis=0)
    err_grad_dcf_swtch = np.sqrt(np.var(grad_dcf_swtch, axis=0, ddof=1) / dcf_swtch.shape[0])

    # # c(r) by fourier inversion of just convolved term for comparision
    # dcf_both = transforms.sq_and_hr_to_cr(r_bins, input_density, block_tcf, r, block_sq, q)
    # avg_dcf_both = np.mean(dcf_both, axis=0)
    # err_dcf_both = np.sqrt(np.var(dcf_both, axis=0, ddof=1) / dcf_both.shape[0])

    ## Evaluate B(r)

    br = np.log(avg_tcf + 1.) + phi - avg_tcf + avg_dcf_swtch
    # bridge = np.log(sw_cav) + avg_dcf_swtch - avg_tcf
    # bridge2 = np.log(sw_cav) + avg_dcf_fft - avg_tcf



    r_peak_dir = r[np.argmax(avg_tcf)]
    r_peak_spl = transforms.spline_max(r, avg_tcf)
    r_peak_err = r[np.argmax(avg_tcf+err_tcf)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Plot phi(r)

    axes[0, 0].plot(r, phi)
    axes[0, 0].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    axes[0, 0].set_ylim([-3, 8])
    axes[0, 0].set_xlabel('r/$\sigma$')
    axes[0, 0].set_ylabel('$\phi(r)$')

    # Plot g(r)

    axes[0, 1].plot(r, avg_tcf + 1, color="tab:blue")
    axes[0, 1].fill_between(r, avg_tcf + err_tcf + 1, avg_tcf - err_tcf + 1, alpha=0.3)
    # axes[0, 1].plot(r, block_tcf.T + 1, alpha=0.1)
    axes[0, 1].plot((r[0],r[-1]), np.ones(2), '--', color="tab:blue")
    axes[0, 1].set_xlabel('r/$\sigma$')
    axes[0, 1].set_ylabel('$g(r)$')

    axes[1,1].plot(r, avg_grad_tcf*r_peak_dir)
    axes[1,1].fill_between(r, (avg_grad_tcf + err_grad_tcf)*r_peak_dir, 
                            (avg_grad_tcf - err_grad_tcf)*r_peak_dir, alpha=0.2)
    axes[1, 1].set_xlabel('r/$\sigma$')
    axes[1, 1].set_ylabel('$g\'(r)$')

    # Plot c(r)

    axes[0, 2].plot(r, avg_dcf_dir, label='$c_{dir}(r)$')
    axes[0, 2].fill_between(r, avg_dcf_dir + err_dcf_dir, avg_dcf_dir - err_dcf_dir, alpha=0.2)

    axes[0, 2].plot(r, avg_dcf_fft, label='$c_{fft}(r)$')
    axes[0, 2].fill_between(r, avg_dcf_fft + err_dcf_fft, avg_dcf_fft - err_dcf_fft, alpha=0.2)

    axes[0, 2].plot(r, avg_dcf_swtch, label='$c_{sw}(r)$')
    axes[0, 2].fill_between(r, avg_dcf_swtch + err_dcf_swtch, avg_dcf_swtch - err_dcf_swtch, alpha=0.2)

    # axes[0, 2].plot(r, avg_dcf_both, label='$c_{indep}(r)$')
    # axes[0, 2].fill_between(r, avg_dcf_both + err_dcf_both, avg_dcf_both - err_dcf_both, alpha=0.2)

    axes[0, 2].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    axes[0, 2].set_xlabel('r/$\sigma$')
    axes[0, 2].set_ylabel('$c(r)$')
    axes[0, 2].legend()

    axes[1,2].plot(r, avg_grad_dcf_swtch*r_peak_dir)
    axes[1,2].fill_between(r, (avg_grad_dcf_swtch + err_grad_dcf_swtch)*r_peak_dir, 
                            (avg_grad_dcf_swtch - err_grad_dcf_swtch)*r_peak_dir, alpha=0.2)
    axes[1, 2].set_xlabel('r/$\sigma$')
    axes[1, 2].set_ylabel('$c\'(r)$')


    # # Plot b(r)

    axes[1, 0].plot(r, br, label='$b_{sw}(r)$')
    axes[1, 0].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    # axes[1, 2].plot(r, bridge, label='$y_{sw}(r)$')
    # # axes[1, 2].plot(llanob[:, 0], llanob[:, 2], label='$y_{llano}(r)$')
    # axes[1, 2].plot(r, bridge2, label='$y_{fft}(r)$')
    # axes[1, 2].set_xlim([0, 3.5])
    axes[1, 0].set_xlabel('r/$\sigma$')
    axes[1, 0].set_ylabel('$B(r)$')
    axes[1, 0].legend(loc=4)

    fig.tight_layout()
    

    # Plot s(q)

    fig_2, axes_2 = plt.subplots(2, figsize=(6,6), sharex=True, 
        gridspec_kw={'height_ratios':[3, 1]})

    axes_2[0].plot(q, avg_sq, linewidth=1, marker='x', label='$S_{dir}(q)$')
    axes_2[0].fill_between(q, avg_sq + err_sq, avg_sq - err_sq, alpha=0.2)

    axes_2[0].plot(q_fft, avg_sq_fft, linewidth=1, marker='o', mfc='none', label='$S_{fft}(q)$')
    axes_2[0].fill_between(q_fft, avg_sq_fft + err_sq_fft, avg_sq_fft - err_sq_fft, alpha=0.2)

    axes_2[0].plot(q_fft, avg_sq_switch, color='g', marker='+', linewidth=1, label='$S_{sw}(q)$')
    axes_2[0].fill_between(q_fft, avg_sq_switch + err_sq_switch, avg_sq_switch - err_sq_switch, alpha=0.2)

    axes_2[0].plot(q_fft, switch, color='r',linewidth=1, marker='*', label='W(q)')
    axes_2[0].set_xlim([0, 12.5])
    # axes_2[1, 0].set_ylim([-.5, 4.0])
    axes_2[0].set_xlabel('$q$')
    axes_2[0].set_ylabel('$S(q), W(q)$')
    axes_2[0].legend()

    axes_2[1].plot(q, (- avg_sq + avg_sq_fft), linewidth=1, marker='x', label='$\Delta S(q)$')
    # axes_2[1].plot(q_sam, (- avg_sq + avg_sq_fft[:len(q_sam)]) , linewidth=1, marker='x', label='$\Delta S(q)$')
    axes_2[1].plot((0,13), (0,0), 'k-.', linewidth=0.5)
    axes_2[1].set_xlabel('$q$')
    axes_2[1].set_ylabel('$\Delta S(q)$')
    axes_2[1].set_xlim([0, 12.5])
    # axes_2[1].set_ylim([-0.1, 0.3])
    # axes_2[1].legend()
    fig_2.tight_layout()

    plt.show()


if __name__ == "__main__":
    opt = parse.parse_input()
    input_path = opt.output
    _, pot_type, pot_number = opt.table.split("_")
    pot_number = re.findall('\d+', pot_number)[-1]
    input_size = opt.box_size
    input_density = opt.rho
    input_temp = opt.temp

    main(input_path, pot_type, pot_number, input_size,
         input_temp, input_density)
