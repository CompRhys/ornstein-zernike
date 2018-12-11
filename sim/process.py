from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms, parse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def main(input_path, input_number, input_bpart, input_cpart, input_density):

    display_figures = True

    dat = np.loadtxt('{}dat_d{}_n{}_p{}.dat'
                     .format(input_path, input_density, input_bpart, input_number))
    rdf = np.loadtxt('{}rdf_d{}_n{}_p{}.dat'
                     .format(input_path, input_density, input_bpart, input_number))
    sq = np.loadtxt('{}sq_d{}_n{}_p{}.dat'
                    .format(input_path, input_density, input_bpart, input_number))
    phi = np.loadtxt('{}phi_d{}_n{}_p{}.dat'
                     .format(input_path, input_density, input_bpart, input_number))
    cav = np.loadtxt('{}cav_d{}_n{}_p{}.dat'
                     .format(input_path, input_density, input_cpart, input_number))
    mu = np.loadtxt('{}mu_d{}_n{}_p{}.dat'
                    .format(input_path, input_density, input_cpart, input_number))

    density = dat[0]
    box_l = dat[1]
    r_bins = dat[2].astype(int)

    r = rdf[:, 0]
    rdf = rdf[:, 1:].T
    phi = phi[:, 1]

    r2 = cav[:, 0]
    dir_cav = np.mean(cav[:, 1:] / np.mean(mu), axis=1)

    q = sq[:, 0]
    sq = sq[:, 1:].T

    # where the switch transition occurs
    q_ind = 5
    r_ind = 5

    # optional 5 point function smoothing
    # rdf = transforms.smooth_function(rdf)
    # sq = transforms.smooth_function(sq)

    # Find block size to remove correlations
    block_size = block.fp_block_length(rdf)
    # print(block_size)

    # RDF
    block_rdf = block.block_data(rdf, block_size)
    avg_rdf = np.mean(block_rdf, axis=0)
    err_rdf = np.sqrt(np.var(block_rdf, axis=0, ddof=1) / block_rdf.shape[0])

    # Structure Factor via Fourier Transform
    sq_fft, q_fft = transforms.hr_to_sq(
        r_bins, density, block_rdf - 1., r, q[0])

    # Correction of low q limit via direct measurement (Frankenstien's)
    spline = interp1d(q, sq, kind='cubic')
    mask = np.where(q_fft < np.max(q))[0]
    q_sam = np.take(q_fft, mask)
    sq_sam = spline(q_sam)
    block_sq = block.block_data(sq_sam, block_size)
    sq_extend = np.pad(
        block_sq, ((0, 0), (0, len(q_fft) - len(q_sam))), 'constant')

    # 0.25 prefactor chosen manually should be ~1/(2*pi) from scaling
    qswitch = 1 / (1 + np.exp(0.25 * box_l *
                              (q_fft - (q_sam[-1] - q_ind * q_sam[0]))))
    sq_switch = qswitch * sq_extend + (1. - qswitch) * sq_fft

    # averages from different approaches
    avg_sq = np.mean(sq_sam, axis=0)
    avg_sq_fft = np.mean(sq_fft, axis=0)
    avg_sq_switch = np.mean(sq_switch, axis=0)
    err_sq_switch = np.sqrt(
        np.var(sq_switch, axis=0, ddof=1) / sq_switch.shape[0])

    # evaluate c(r) from corrected structure factor
    cr_swtch, r_swtch = transforms.sq_to_cr(r_bins, density, sq_switch, q_fft)
    avg_cr_swtch = np.mean(cr_swtch, axis=0)
    err_cr_swtch = np.sqrt(
        np.var(cr_swtch, axis=0, ddof=1) / cr_swtch.shape[0])

    # c(r) by fourier inversion for comparision
    cr_fft = transforms.hr_to_cr(r_bins, density, block_rdf - 1, r)
    avg_cr_fft = np.mean(cr_fft, axis=0)
    err_cr_fft = np.sqrt(np.var(cr_fft, axis=0, ddof=1) / cr_fft.shape[0])

    dcav_extend = np.pad(dir_cav, (0, len(r) - len(r2)), 'constant')
    rs_start = 1.04
    rs_end = 1.16
    rs = np.arange(0, rs_end - rs_start, r[0])
    rswitch = 1. / (1. + np.exp(8 * box_l * (rs - rs.mean())))
    rswitch = np.pad(
        rswitch, (int(rs_start / r[0]), 0), 'constant', constant_values=(1))
    rswitch = np.pad(
        rswitch, (0, int((r[-1] - rs_end) / r[0]) + 1), 'constant', constant_values=(0))
    bulk_cav = np.exp(phi) * avg_rdf

    print(bulk_cav.shape, dcav_extend.shape)
    sw_cav = rswitch * dcav_extend + (1 - rswitch) * np.nan_to_num(bulk_cav)

    bridge = np.log(sw_cav) + avg_cr_swtch - avg_rdf + 1
    bridge2 = np.log(sw_cav) + avg_cr_fft - avg_rdf + 1

    if display_figures:
        plot_figure(r, avg_rdf, err_rdf, avg_cr_fft, err_cr_fft, avg_cr_swtch, err_cr_swtch, phi,
                    q_sam, q_fft, avg_sq, avg_sq_fft, avg_sq_switch, err_sq_switch, qswitch, r2, bulk_cav,
                    sw_cav, dir_cav, rswitch, bridge, bridge2)


def save_data(r, avg_rdf, err_rdf, avg_cr_swtch, err_cr_swtch, phi, bridge):
    output = np.column_stack(
        (r, avg_rdf, err_rdf, avg_cr_swtch, err_cr_swtch, phi, bridge))
    np.savetxt(path, output)


def plot_figure(r, avg_rdf, err_rdf, avg_cr_fft, err_cr_fft, avg_cr_swtch, err_cr_swtch, phi,
                q_sam, q_fft, avg_sq, avg_sq_fft, avg_sq_switch, err_sq_switch, qswitch, r2, bulk_cav,
                sw_cav, dir_cav, rswitch, bridge, bridge2):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # plot phi(r)

    axes[0, 0].plot(r, phi)
    axes[0, 0].set_ylim([-3, 8])
    axes[0, 0].set_xlabel('r/$\sigma$')
    axes[0, 0].set_ylabel('$\phi(r)$')

    # plot g(r)

    axes[0, 1].errorbar(r, avg_rdf, err_rdf,
                        linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
    axes[0, 1].plot(r, np.ones(len(r)), 'C0--', linewidth=1.0)
    axes[0, 1].set_xlabel('r/$\sigma$')
    axes[0, 1].set_ylabel('$g(r)$')

    # plot c(r)

    axes[0, 2].errorbar(r, avg_cr_fft, err_cr_fft, color='C0',
                        linewidth=1, errorevery=20, ecolor='C0',
                        elinewidth=1, label='$c_{fft}(r)$')
    axes[0, 2].errorbar(r, avg_cr_swtch, err_cr_swtch, color='g',
                        linewidth=1, errorevery=20, ecolor='g',
                        elinewidth=1, label='$c_{sw}(r)$')
    axes[0, 2].plot(r, np.zeros(len(r)), 'C0--', linewidth=1.0)
    axes[0, 2].set_xlabel('r/$\sigma$')
    axes[0, 2].set_ylabel('$c(r)$')
    axes[0, 2].legend()



    # Plot s(q)

    axes[1, 0].plot(q_sam, avg_sq, linewidth=1,
                    marker='x', label='$S_{dir}(q)$')
    axes[1, 0].plot(q_fft, avg_sq_fft, linewidth=1,
                    marker='x', label='$S_{fft}(q)$')
    axes[1, 0].errorbar(q_fft, avg_sq_switch, err_sq_switch,
                        color='g', marker='x', linewidth=1,
                        errorevery=20, ecolor='g', elinewidth=1,
                        label='$S_{avg}(q)$')
    axes[1, 0].plot(q_fft, qswitch, color='r',
                    linewidth=1, marker='x', label='W(q)')
    axes[1, 0].set_xlim([0, 12.5])
    axes[1, 0].set_ylim([0, 2.0])
    axes[1, 0].set_xlabel('$q$')
    axes[1, 0].set_ylabel('$S(q), W(q)$')
    axes[1, 0].legend()

    # plot y(r)

    axes[1, 1].plot(r, np.arcsinh(bulk_cav), label='$y_{dir}(r)$')
    axes[1, 1].plot(r2, np.arcsinh(dir_cav), label='$y_{cav}(r)$')
    axes[1, 1].plot(r, np.arcsinh(sw_cav), label='$y_{sw}(r)$')
    axes[1, 1].plot(r, rswitch, label='W(r)')
    # axes[1, 1].set_xlim([0, 2])
    axes[1, 1].set_xlabel('r/$\sigma$')
    axes[1, 1].set_ylabel('$y(r)$')
    axes[1, 1].legend()

    # plot b(r)

    # llano = np.array((-6.8757, -6.5826, -6.1928, -5.7682, -5.3394, -4.9202, -4.5109, -4.1127, -3.7278, -3.3629,
    #                   -2.9926, -2.6308, -2.292, -1.9863, -1.6845, -1.4042, -1.1465, -0.8964, -0.692,
    #                   -0.5205, -0.3708, -0.25, -0.1581, -0.0912, -0.0477, -0.0242, -0.0173, -0.0122, -0.0154,
    #                   -0.0239, -0.0362, -0.0405, -0.0373, -0.0395, -0.0251, -0.0193, -0.0085, -0.0011, -0.0043,
    #                   -0.0091, 0.0196, 0.0192, 0.0217, 0.0211, 0.0164, 0.0148, 0.0144, 0.0123,-0.0176, -0.0223))
    # rllano = np.arange(0.05, 2.55, 0.05)
    # axes[1, 2].plot(rllano, llano)

    axes[1, 2].plot(r, bridge, label='$y_{sw}(r)$')
    axes[1, 2].plot(r, bridge2, label='$y_{fft}(r)$')
    # axes[1, 2].set_xlim([0, 2])
    axes[1, 2].set_xlabel('r/$\sigma$')
    axes[1, 2].set_ylabel('$B(r)$')
    axes[1, 2].legend(loc=4)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    opt = parse.parse_input()
    input_path = opt.output
    input_number = re.findall('\d+', opt.table)[0]
    input_bpart = opt.bulk_part
    input_cpart = opt.cav_part
    input_density = opt.rho

    main(input_path, input_number, input_bpart, input_cpart, input_density)
