from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def main(input_path, input_number):

    save = False
    display_figures = True

    dat = np.loadtxt(input_path+'dat_'+input_number+'.dat')
    rdf = np.loadtxt(input_path+'rdf_'+input_number+'.dat')
    sq = np.loadtxt(input_path+'sq_'+input_number+'.dat')

    density = dat[0]
    box_l = dat[1]
    r_bins = dat[2].astype(int)

    r = rdf[:,0]
    rdf = rdf[:,1:].T

    q = sq[:,0]
    sq = sq[:,1:].T

    # mask = np.where(q % q[0] < 1e-14)[0]
    # q = np.take(q, mask)
    # sq = np.take(sq, mask, axis = 1)

    # set this somehow
    sw_ind = 5

    # rdf = transforms.smooth_function(rdf)
    sq = transforms.smooth_function(sq)

    # Find block size to remove correlations
    block_size = block.fp_block_length(rdf)
    # print(block_size)

    # RDF
    block_rdf = block.block_data(rdf, block_size)
    avg_rdf = np.mean(block_rdf, axis=0)
    err_rdf = np.sqrt(np.var(block_rdf, axis=0, ddof=1) / block_rdf.shape[0])

    # Structure Factor via Fourier Transform
    sq_fft, q_fft = transforms.hr_to_sq(r_bins, density, block_rdf - 1., r, q[0])

    # Correction of low q limit via direct measurement (Frankenstien's)
    spline = interp1d(q, sq, kind='cubic')
    mask = np.where(q_fft < np.max(q))[0]
    q_sam = np.take(q_fft, mask)
    sq_sam = spline(q_sam)
    block_sq = block.block_data(sq_sam, block_size)
    sq_extend = np.pad(block_sq, ((0, 0), (0, len(q_fft) - len(q_sam))), 'constant')

    # 0.25 prefactor chosen manually should be ~1/(2*pi) from scaling
    switch = 1. / (1. + np.exp(0.25 * box_l * (q_sam - (q_sam[-1]-sw_ind*q_sam[0]))))
    switch_extend = np.pad(switch, (0, len(q_fft) - len(q_sam)), 'constant')

    sq_switch = switch_extend * sq_extend + (1. - switch_extend) * sq_fft

    # averages from different approaches
    avg_sq = np.mean(sq_sam, axis=0)
    avg_sq_fft = np.mean(sq_fft, axis=0)
    avg_sq_switch = np.mean(sq_switch, axis=0)
    err_sq_switch = np.sqrt(np.var(sq_switch, axis=0, ddof=1) / sq_switch.shape[0])

    print(err_sq_switch)

    # evaluate c(r) from corrected structure factor
    cr_swtch, r_swtch = transforms.sq_to_cr(r_bins, density, sq_switch, q_fft)
    avg_cr_swtch = np.mean(cr_swtch, axis=0)
    err_cr_swtch = np.sqrt(np.var(cr_swtch, axis=0, ddof=1) / cr_swtch.shape[0])

    # c(r) by fourier inversion for comparision

    cr_fft = transforms.hr_to_cr(r_bins, density, block_rdf - 1, r)
    avg_cr_fft = np.mean(cr_fft, axis=0)
    err_cr_fft = np.sqrt(np.var(cr_fft, axis=0, ddof=1) / cr_fft.shape[0])

    if save:
        save_output()

    if display_figures:
        fig, axes = plt.subplots(2, figsize=(10, 10))
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

        plt.figure()
        plt.plot(q_sam, avg_sq, linewidth=1, marker='x', label='$S_{dir}(q)$')
        plt.plot(q_fft, avg_sq_fft, linewidth=1, marker='x', label='$S_{fft}(q)$')
        plt.errorbar(q_fft, avg_sq_switch, err_sq_switch, color='g', marker='x',
                     linewidth=1, errorevery=20, ecolor='g', elinewidth=1, label='$S_{avg}(q)$')
        plt.plot(q_fft, switch_extend, color='r',
                 linewidth=1, marker='x', label='W(q)')
        plt.xlim([0, 12.5])
        plt.ylim([0, 2.0])
        plt.xlabel('$q$')
        plt.ylabel('$S(q), W(q)$')
        plt.legend()
        plt.tight_layout()

        plt.show()

def save_output():
    pass


if __name__ == "__main__":
    input_path = sys.argv[1]
    input_number = sys.argv[2]
    main(input_path, input_number)
