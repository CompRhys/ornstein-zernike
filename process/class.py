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
    cav = np.loadtxt('{}cav_d{}_n{}_t{}_p{}.dat'.format(
        input_path, input_density, input_cpart, input_temp, input_number))
    mu = np.loadtxt('{}mu_d{}_n{}_t{}_p{}.dat'.format(
        input_path, input_density, input_cpart, input_temp, input_number))

    # llanob = np.genfromtxt('sim/core/llanob.dat', delimiter=',', dtype=float)
    # llanoy = np.genfromtxt('sim/core/llanoy.dat', delimiter=',', dtype=float)

    # print(llanoy.shape)

    r = rdf[0, :]
    r_bins = len(r)
    rdf = rdf[1:, :]
    phi = phi[1, :]

    r2 = cav[0, :]
    cav = cav[1:, :]
    # cav = transforms.smooth_function(cav, 1)
    dir_cav = np.mean(cav.T / mu, axis=1)
    dirm_cav = np.mean(cav, axis=0) / np.mean(mu)
    err_cav = np.std(cav.T / mu, axis=1, ddof=1) / np.sqrt(cav.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].plot(r2, np.arcsinh(cav.T / mu))
    axes[1, 0].plot(r2, np.arcsinh(cav.T / np.mean(mu)))
    axes[0, 1].hist(mu, 'auto')
    axes[1, 1].plot(r2, (dir_cav))
    axes[1, 1].plot(r2, (dirm_cav))
    # axes[1, 1].plot(llanoy[:, 0], np.exp(llanoy[:, 2]))

    fig.tight_layout()

    q = sq[0, :]
    sq = sq[1:, :]

    # where the switch transition occurs
    q_ind = 7
    r_ind = 5

    # optional 5 point function smoothing
    # rdf = transforms.smooth_function(rdf)
    # sq = transforms.smooth_function(sq)

    # Find block size to remove correlations
    # block_size = block.fp_block_length(rdf)
    # block_size = 32

    # RDF
    block_rdf = rdf
    # block_rdf = block.block_data(rdf, block_size)
    avg_rdf = np.mean(block_rdf, axis=0)
    err_rdf = np.sqrt(np.var(block_rdf, axis=0, ddof=1) / block_rdf.shape[0])

    # Structure Factor via Fourier Transform
    sq_fft, q_fft = transforms.hr_to_sq(
        r_bins, input_density, block_rdf - 1., r)

    # Correction of low q limit via direct measurement (Frankenstien's)
    spline = interp1d(q, sq, kind='cubic')
    mask = np.where(q_fft < np.max(q))[0]
    q_sam = np.take(q_fft, mask)
    sq_sam = spline(q_sam)

    block_sq = sq_sam
    # block_sq = block.block_data(sq_sam, block_size)
    sq_extend = np.pad(
        block_sq, ((0, 0), (0, len(q_fft) - len(q_sam))), 'constant')

    # 0.25 prefactor chosen manually should be ~1/(2*pi) from scaling
    # qswitch = 1 / (1 + np.exp(0.25 * box_l * (q_fft - (q_sam[-1] - q_ind * q_sam[0]))))
    switch = (1 + np.cos(np.pi * q_sam[:q_ind] / q_sam[q_ind])) * .5
    before = 4
    after = len(q_fft) - q_ind - before
    switch = np.pad(switch, (before, 0), 'constant', constant_values=(1))
    qswitch = np.pad(switch, (0, after), 'constant', constant_values=(0))
    sq_switch = qswitch * sq_extend + (1. - qswitch) * sq_fft

    # averages from different approaches
    avg_sq = np.mean(block_sq, axis=0)
    err_sq = np.sqrt(
        np.var(block_sq, axis=0, ddof=1) / block_sq.shape[0])
    avg_sq_fft = np.mean(sq_fft, axis=0)
    err_sq_fft = np.sqrt(
        np.var(sq_fft, axis=0, ddof=1) / sq_fft.shape[0])
    avg_sq_switch = np.mean(sq_switch, axis=0)
    err_sq_switch = np.sqrt(
        np.var(sq_switch, axis=0, ddof=1) / sq_switch.shape[0])

    # evaluate c(r) from corrected structure factor
    cr_swtch, r_swtch = transforms.sq_to_cr(
        r_bins, input_density, sq_switch, q_fft)
    avg_cr_swtch = np.mean(cr_swtch, axis=0)
    err_cr_swtch = np.sqrt(
        np.var(cr_swtch, axis=0, ddof=1) / cr_swtch.shape[0])

    # c(r) by fourier inversion for comparision
    cr_fft = transforms.hr_to_cr(r_bins, input_density, block_rdf - 1, r)
    avg_cr_fft = np.mean(cr_fft, axis=0)
    err_cr_fft = np.sqrt(np.var(cr_fft, axis=0, ddof=1) / cr_fft.shape[0])

    # todo: look at this in detail to ensure we are matching correctly
    dcav_extend = np.pad(dirm_cav, (0, len(r) - len(r2)), 'constant')
    rs_start = 0.9
    rs_end = 1.0
    rs = np.arange(0, rs_end - rs_start, r[0])
    before = np.floor(rs_start / r[0]).astype(int)
    after = np.ceil((r[-1] - rs_end) / r[0]).astype(int)
    # rswitch = 1. / (1. + np.exp(20 * (rs - rs.mean())))
    rswitch = (1 + np.cos(np.pi * rs / rs[-1])) * .5
    rswitch = np.pad(rswitch, (before, 0), 'constant', constant_values=(1))
    rswitch = np.pad(rswitch, (0, after), 'constant', constant_values=(0))
    bulk_cav = np.exp(phi) * avg_rdf
    sw_cav = rswitch * dcav_extend + (1 - rswitch) * np.nan_to_num(bulk_cav)

    bridge = np.log(sw_cav) + avg_cr_swtch - avg_rdf + 1
    bridge2 = np.log(sw_cav) + avg_cr_fft - avg_rdf + 1

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    rev_rdf, tr = transforms.sq_to_hr(r_bins, input_density, sq_switch, q_fft)
    tesy = np.mean(rev_rdf, axis=0)

    # plot phi(r)

    axes[0, 0].plot(r, phi)
    axes[0, 0].set_ylim([-3, 8])
    axes[0, 0].set_xlabel('r/$\sigma$')
    axes[0, 0].set_ylabel('$\phi(r)$')

    # plot g(r)

    # axes[0, 1].plot(r, block_rdf.T)
    # axes[0, 1].plot(tr, tesy + 1)
    # axes[0, 1].plot(tr, (- avg_rdf + tesy + 1) * 10 + 1)
    axes[0, 1].plot(r, avg_rdf, color="tab:blue")
    # axes[0, 1].fill_between(r, avg_rdf + err_rdf * 100, avg_rdf - err_rdf * 100,
    #                         alpha=0.3)
    # axes[0, 1].plot(r, np.ones(len(r)), '--', linewidth=1.0)
    axes[0, 1].set_xlabel('r/$\sigma$')
    axes[0, 1].set_ylabel('$g(r)$')

    # plot c(r)

    axes[0, 2].plot(r, avg_cr_swtch, label='$c_{sw}(r)$')
    axes[0, 2].fill_between(r, avg_cr_swtch + err_cr_swtch * 100,
                            avg_cr_swtch - err_cr_swtch * 100, alpha=0.2)
    axes[0, 2].plot(r, avg_cr_fft, label='$c_{fft}(r)$')
    # axes[0, 2].fill_between(r, avg_cr_fft + err_cr_fft * 100,
    #                         avg_cr_fft - err_cr_fft * 100, alpha=0.2)
    axes[0, 2].plot(r, np.zeros(len(r)), 'C0--', linewidth=1.0)
    axes[0, 2].set_xlabel('r/$\sigma$')
    axes[0, 2].set_ylabel('$c(r)$')
    axes[0, 2].legend()

    # Plot s(q)

    axes[1, 0].plot(q_sam, avg_sq, linewidth=1,
                    marker='x', label='$S_{dir}(q)$')
    axes[1, 0].plot(q_fft, avg_sq_fft, linewidth=1,
                    marker='x', label='$S_{fft}(q)$')
    axes[1, 0].plot(q_fft, avg_sq_switch,
                    color='g', marker='x', linewidth=1,
                    label='$S_{avg}(q)$')
    axes[1, 0].fill_between(q_sam, avg_sq + err_sq * 10,
                            avg_sq - err_sq * 10, alpha=0.2)
    axes[1, 0].fill_between(q_fft, avg_sq_fft + err_sq_fft * 10,
                            avg_sq_fft - err_sq_fft * 10, alpha=0.2)
    axes[1, 0].fill_between(q_fft, avg_sq_switch + err_sq_switch * 10,
                            avg_sq_switch - err_sq_switch * 10, alpha=0.2)
    axes[1, 0].plot(q_fft, qswitch, color='r',
                    linewidth=1, marker='x', label='W(q)')
    # axes[0, 1].plot(r, avg_rdf)
    axes[1, 0].set_xlim([0, 12.5])
    # axes[1, 0].set_ylim([-.5, 2.0])
    axes[1, 0].set_xlabel('$q$')
    axes[1, 0].set_ylabel('$S(q), W(q)$')
    axes[1, 0].legend()

    # plot y(r)

    axes[1, 1].plot(r, np.arcsinh(bulk_cav), label='$y_{dir}(r)$')
    axes[1, 1].plot(r2, np.arcsinh(dir_cav), label='$y_{cav}(r)$')
    axes[1, 1].fill_between(r2, np.arcsinh(dir_cav + err_cav),
                            np.arcsinh(dir_cav - err_cav), alpha=0.1)
    axes[1, 1].plot(r2, np.arcsinh(dirm_cav), label='$y_{mean}(r)$')
    # axes[1, 1].plot(llanoy[:, 0], np.arcsinh(
    #     np.exp(llanoy[:, 3])), label='$y_{llano}(r)$')
    axes[1, 1].plot(r, np.arcsinh(sw_cav), label='$y_{sw}(r)$')
    axes[1, 1].plot(r, rswitch, label='W(r)')
    # axes[1, 1].set_ylim([0,5])
    axes[1, 1].set_xlabel('r/$\sigma$')
    axes[1, 1].set_ylabel('$y(r)$')
    axes[1, 1].legend()

    # plot b(r)

    axes[1, 2].plot(r, bridge, label='$y_{sw}(r)$')
    # axes[1, 2].plot(llanob[:, 0], llanob[:, 2], label='$y_{llano}(r)$')
    axes[1, 2].plot(r, bridge2, label='$y_{fft}(r)$')
    # axes[1, 2].set_xlim([0, 2])
    axes[1, 2].set_xlabel('r/$\sigma$')
    axes[1, 2].set_ylabel('$B(r)$')
    axes[1, 2].legend(loc=4)

    fig.tight_layout()

    f, axarr = plt.subplots(2, figsize=(6,6), sharex=True, 
        gridspec_kw={'height_ratios':[3, 1]})

    axarr[0].plot(q_fft, avg_sq_fft, linewidth=1,
                  marker='x', label='$S_{fft}(q)$')
    axarr[0].plot(q_sam, avg_sq, linewidth=1,
                  marker='x', label='$S_{dir}(q)$')
    axarr[0].plot(q_fft, avg_sq_switch,
                  color='g', marker='x', linewidth=1,
                  label='$S_{hybrid}(q)$')
    axarr[0].plot(q_fft, qswitch, color='r',
                    linewidth=1, marker='x', label='W(q)')
    axarr[0].set_ylabel('$S(q), W(q)$')
    axarr[0].legend()
    axarr[1].plot(q_sam, (- avg_sq + avg_sq_fft[:len(q_sam)]) , 
        linewidth=1, marker='x', label='$\Delta S(q)$')
    axarr[1].plot((0,13), (0,0), 'k-.', linewidth=0.5)
    axarr[1].set_xlabel('$q$')
    axarr[1].set_ylabel('$\Delta S(q)$')
    axarr[1].set_xlim([0, 12.5])
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
