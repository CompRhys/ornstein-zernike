from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms, parse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def main(input_path, input_number, input_bpart, input_cpart, 
    input_temp, input_density):

    display_figures = True

    rdf = np.loadtxt('{}rdf_d{}_n{}_t{}_p{}.dat'.format(
        input_path, input_density, input_bpart, input_temp, input_number))
    sq = np.loadtxt('{}sq_d{}_n{}_t{}_p{}.dat'.format(
        input_path, input_density, input_bpart, input_temp, input_number))
    phi = np.loadtxt('{}phi_p{}.dat'.format(
        input_path, input_number))
    cav = np.loadtxt('{}cav_d{}_n{}_t{}_p{}.dat'.format(
        input_path, input_density, input_cpart, input_temp, input_number))
    mu = np.loadtxt('{}mu_d{}_n{}_t{}_p{}.dat'.format(
        input_path, input_density, input_cpart, input_temp, input_number))

    r = rdf[0, :]
    r_bins = len(r)
    rdf = rdf[1:, :]
    phi = phi[1, :]



    r2 = cav[0, :]
    cav = cav[1:, :]
    cav = transforms.smooth_function(cav, 1)
    dir_cav = np.mean(cav.T / mu, axis=1)
    dirm_cav = np.mean(cav.T / np.mean(mu), axis=1)
    err_cav = np.std(cav.T / mu, axis=1, ddof=1)/np.sqrt(cav.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].plot(r2, np.arcsinh(cav.T / mu))
    axes[0, 1].plot(r2, np.arcsinh(cav.T/ np.mean(mu)))
    axes[1, 0].hist(mu, 'auto')
    axes[1, 1].plot(r2, np.arcsinh(dir_cav))
    axes[1, 1].plot(r2, np.arcsinh(dirm_cav))

    fig.tight_layout()
    plt.show()

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
    block_size = 128

    # RDF
    # block_rdf = rdf
    block_rdf = block.block_data(rdf, block_size)
    avg_rdf = np.mean(block_rdf, axis=0)
    err_rdf = np.sqrt(np.var(block_rdf, axis=0, ddof=1) / block_rdf.shape[0])

    # Structure Factor via Fourier Transform
    sq_fft, q_fft = transforms.hr_to_sq(
        r_bins, input_density, block_rdf - 1., r, q[0])

    # Correction of low q limit via direct measurement (Frankenstien's)
    spline = interp1d(q, sq, kind='cubic')
    mask = np.where(q_fft < np.max(q))[0]
    q_sam = np.take(q_fft, mask)
    sq_sam = spline(q_sam)

    block_sq = block.block_data(sq_sam, block_size)
    sq_extend = np.pad(
        block_sq, ((0, 0), (0, len(q_fft) - len(q_sam))), 'constant')

    # 0.25 prefactor chosen manually should be ~1/(2*pi) from scaling
    # qswitch = 1 / (1 + np.exp(0.25 * box_l * (q_fft - (q_sam[-1] - q_ind * q_sam[0]))))
    coswitch = (1 + np.cos(np.pi * q_sam[:q_ind] / q_sam[q_ind])) * .5
    qswitch = np.pad(coswitch, (0, len(q_fft) - q_ind), 'constant')
    sq_switch = qswitch * sq_extend + (1. - qswitch) * sq_fft

    # averages from different approaches
    avg_sq = np.mean(sq_sam, axis=0)
    avg_sq_fft = np.mean(sq_fft, axis=0)
    avg_sq_switch = np.mean(sq_switch, axis=0)
    err_sq_switch = np.sqrt(
        np.var(sq_switch, axis=0, ddof=1) / sq_switch.shape[0])

    # evaluate c(r) from corrected structure factor
    cr_swtch, r_swtch = transforms.sq_to_cr(r_bins, input_density, sq_switch, q_fft)
    avg_cr_swtch = np.mean(cr_swtch, axis=0)
    err_cr_swtch = np.sqrt(
        np.var(cr_swtch, axis=0, ddof=1) / cr_swtch.shape[0])

    # c(r) by fourier inversion for comparision
    cr_fft = transforms.hr_to_cr(r_bins, input_density, block_rdf - 1, r)
    avg_cr_fft = np.mean(cr_fft, axis=0)
    err_cr_fft = np.sqrt(np.var(cr_fft, axis=0, ddof=1) / cr_fft.shape[0])

    # todo: look at this in detail to ensure we are matching correctly
    dcav_extend = np.pad(dir_cav, (0, len(r) - len(r2) + 1), 'constant')
    rs_start = 1.0
    rs_end = r2[-1]
    rs = np.arange(0, rs_end - rs_start, r[0])
    before = int(rs_start / r[0])
    after = int((r[-1] - rs_end) / r[0])
    # rswitch = 1. / (1. + np.exp(20 * (rs - rs.mean())))
    rswitch = (1 + np.cos(np.pi * rs / rs[-1])) * .5
    rswitch = np.pad(rswitch, (before, 0), 'constant', constant_values=(1))
    rswitch = np.pad(rswitch, (0, after), 'constant', constant_values=(0))
    bulk_cav = np.exp(phi) * avg_rdf
    print(rswitch.shape, dcav_extend.shape, bulk_cav.shape)

    sw_cav = rswitch * dcav_extend[1:] + (1 - rswitch) * np.nan_to_num(bulk_cav)

    bridge = np.log(sw_cav) + avg_cr_swtch - avg_rdf + 1
    bridge2 = np.log(sw_cav) + avg_cr_fft - avg_rdf + 1

    if display_figures:
        plot_figure(r, block_rdf, avg_rdf, err_rdf, avg_cr_fft, err_cr_fft, avg_cr_swtch, err_cr_swtch, phi,
                    q_sam, q_fft, avg_sq, avg_sq_fft, avg_sq_switch, err_sq_switch, qswitch, r2, rs, bulk_cav,
                    sw_cav, dir_cav, rswitch, bridge, bridge2, err_cav, dirm_cav)


def save_data(r, avg_rdf, err_rdf, avg_cr_swtch, err_cr_swtch, phi, bridge):
    output = np.column_stack(
        (r, avg_rdf, err_rdf, avg_cr_swtch, err_cr_swtch, phi, bridge))
    np.savetxt(path, output)


def plot_figure(r, rdf, avg_rdf, err_rdf, avg_cr_fft, err_cr_fft, avg_cr_swtch, err_cr_swtch, phi,
                q_sam, q_fft, avg_sq, avg_sq_fft, avg_sq_switch, err_sq_switch, qswitch, r2, rs, bulk_cav,
                sw_cav, dir_cav, rswitch, bridge, bridge2, err_cav, dirm_cav):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # plot phi(r)

    axes[0, 0].plot(r, phi)
    axes[0, 0].set_ylim([-3, 8])
    axes[0, 0].set_xlabel('r/$\sigma$')
    axes[0, 0].set_ylabel('$\phi(r)$')

    # plot g(r)

    axes[0, 1].plot(r, rdf.T)
    axes[0, 1].errorbar(r, avg_rdf, err_rdf, color='r',
                        linewidth=2, errorevery=20, ecolor='r', elinewidth=1)
    axes[0, 1].plot(r, np.ones(len(r)), 'C0--', linewidth=1.0)
    axes[0, 1].set_xlabel('r/$\sigma$')
    axes[0, 1].set_ylabel('$g(r)$')

    # plot c(r)

    axes[0, 2].errorbar(r, avg_cr_fft, err_cr_fft, color='C0',
                        linewidth=1, errorevery=20, ecolor='C0',
                        elinewidth=1, label='$c_{fft}(r)$')
    axes[0, 2].errorbar(r, avg_cr_swtch, err_cr_swtch, color='C1',
                        linewidth=1, errorevery=20, ecolor='C1',
                        elinewidth=1, label='$c_{sw}(r)$')
    axes[0, 2].plot(r, np.zeros(len(r)), 'C0--', linewidth=1.0)
    axes[0, 2].set_xlabel('r/$\sigma$')
    axes[0, 2].set_ylabel('$c(r)$')
    axes[0, 2].legend()

    # Plot s(q)
    axes[1, 0].plot(q_sam, (avg_sq - avg_sq_fft[:len(q_sam)]) * 10, linewidth=1,
                    marker='x', label='$S_{dir}(q)$')
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
    axes[1, 0].set_ylim([-.5, 2.0])
    axes[1, 0].set_xlabel('$q$')
    axes[1, 0].set_ylabel('$S(q), W(q)$')
    axes[1, 0].legend()

    # plot y(r)

    # llano = np.array([12.78151517,
                      # 11.94843133,
                      # 9.93138555,
                      # 8.424128258,
                      # 7.049805304,
                      # 5.629946841,
                      # 4.747413769,
                      # 4.067383833,
                      # 3.484065991,
                      # 2.946446889,
                      # 2.619281386,
                      # 2.280284008,
                      # 2.028912631,
                      # 1.843931543,
                      # 1.568312185,
                      # 1.486315145,
                      # 1.3366948,
                      # 1.255582992,
                      # 1.184712347,
                      # 1.075300331,
                      # 1.030454534,
                      # 0.966281577,
                      # 0.942801045,
                      # 0.89906495,
                      # 0.864157703,
                      # 0.843749187,
                      # 0.817013223,
                      # 0.811557531,
                      # 0.794771998,
                      # 0.791757596,
                      # 0.788912393,
                      # 0.800034842,
                      # 0.801556353,
                      # 0.802599054,
                      # 0.821519175,
                      # 0.85018611,
                      # 0.868315631,
                      # 0.889318358,
                      # 0.940446986,
                      # 0.9604052,
                      # 0.989653893,
                      # 1.000100005,
                      # 1.017247042,
                      # 1.023778301,
                      # 1.025827906,
                      # 1.023675928,
                      # 1.016941914,
                      # 1.020813645,
                      # 1.010555318,
                      # 1.004912025])
    # rllano = np.arange(0.05, 2.55, 0.05)
    # axes[1, 1].plot(rllano, np.arcsinh(llano), label='$y_{llano}(r)$')
    # axes[1, 1].plot(rllano, llano, label='$y_{sw}(r)$')

    axes[1, 1].plot(r, np.arcsinh(bulk_cav), label='$y_{dir}(r)$')
    axes[1, 1].plot(r2, np.arcsinh(dir_cav), label='$y_{cav}(r)$')
    axes[1, 1].plot(r2, np.arcsinh(dirm_cav), label='$y_{mean}(r)$')
    # axes[1, 1].plot(r, np.arcsinh(sw_cav), label='$y_{sw}(r)$')
    # axes[1, 1].plot(r, rswitch, label='W(r)')
    # axes[1, 1].set_ylim([0,5])
    axes[1, 1].set_xlabel('r/$\sigma$')
    axes[1, 1].set_ylabel('$y(r)$')
    axes[1, 1].legend()

    # plot b(r)

    # llano = np.array((-6.8757,
                      # -6.5826,
                      # -6.1928,
                      # -5.7682,
                      # -5.3394,
                      # -4.9202,
                      # -4.5109,
                      # -4.1127,
                      # -3.7278,
                      # -3.3629,
                      # -2.9926,
                      # -2.6308,
                      # -2.292,
                      # -1.9863,
                      # -1.6845,
                      # -1.4042,
                      # -1.1465,
                      # -0.8964,
                      # -0.692,
                      # -0.5205,
                      # -0.3708,
                      # -0.25,
                      # -0.1581,
                      # -0.0912,
                      # -0.0477,
                      # -0.0242,
                      # -0.0173,
                      # -0.0122,
                      # -0.0154,
                      # -0.0239,
                      # -0.0362,
                      # -0.0405,
                      # -0.0373,
                      # -0.0395,
                      # -0.0251,
                      # -0.0193,
                      # -0.0085,
                      # -0.0011,
                      # -0.0043,
                      # -0.0091,
                      # 0.0196,
                      # 0.0192,
                      # 0.0217,
                      # 0.0211,
                      # 0.0164,
                      # 0.0148,
                      # 0.0144,
                      # 0.0123,
                      # -0.0176,
                      # -0.0223))
    # llano = np.array((-0.8924,
    #                   -0.8692,
    #                   -0.8132,
    #                   -0.7585,
    #                   -0.6901,
    #                   -0.6251,
    #                   -0.5703,
    #                   -0.5032,
    #                   -0.4370,
    #                   -0.4051,
    #                   -0.3345,
    #                   -0.3114,
    #                   -0.2711,
    #                   -0.2134,
    #                   -0.2247,
    #                   -0.1658,
    #                   -0.1502,
    #                   -0.1172,
    #                   -0.0731,
    #                   -0.0734,
    #                   -0.0631,
    #                   -0.0527,
    #                   -0.0164,
    #                   -0.0204,
    #                   -0.0195,
    #                   -0.0016,
    #                   -0.0156,
    #                   -0.0154,
    #                   -0.0159,
    #                   -0.0288,
    #                   -0.0418,
    #                   -0.0155,
    #                   -0.0169,
    #                   -0.0209,
    #                   -0.0192,
    #                   -0.0177,
    #                   -0.0293,
    #                   -0.0296,
    #                   -0.0027,
    #                   0.0014,
    #                   -0.0012,
    #                   -0.0073,
    #                   0.0003,
    #                   -0.0019,
    #                   0.0080,
    #                   0.0085,
    #                   0.0027,
    #                   0.0123,
    #                   0.0057,
    #                   0.0079))
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
    input_temp = opt.temp

    main(input_path, input_number, input_bpart, input_cpart, 
        input_temp, input_density)
