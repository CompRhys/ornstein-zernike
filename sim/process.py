from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms, parse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class processing:

    def __init__(self):
        self.block_size = 128

        self.rho = []
        self.r_bins = []
        self.r = []
        self.r_ = []

        self.raw_rdf = []
        self.phi = []

        self.raw_cav = []
        self.mu = []

        self.q = []
        self.q_ = []
        self.raw_sq = []

    def load_data(self, path, number, bpart, cpart, temp, density):
        # load the data

        rdf = np.loadtxt('{}rdf_d{}_n{}_t{}_p{}.dat'.format(
            path, density, bpart, temp, number))
        sq = np.loadtxt('{}sq_d{}_n{}_t{}_p{}.dat'.format(
            path, density, bpart, temp, number))
        phi = np.loadtxt('{}phi_p{}.dat'.format(
            path, number))
        cav = np.loadtxt('{}cav_d{}_n{}_t{}_p{}.dat'.format(
            path, density, cpart, temp, number))
        mu = np.loadtxt('{}mu_d{}_n{}_t{}_p{}.dat'.format(
            path, density, cpart, temp, number))

        # extract the data
        self.rho = density
        self.r_bins = len(rdf[0, :])
        self.r = rdf[0, :]
        self.raw_rdf = rdf[1:, :]
        self.phi = phi[1, :]

        self.r_ = cav[0, :]
        self.raw_cav = cav[1:, :]
        self.mu = mu

        self.q_ = sq[0, :]
        self.raw_sq = sq[1:, :]

    def rdf(self):

        blocked = block.block_data(self.raw_rdf, self.block_size)
        avg_rdf = np.mean(blocked, axis=0)
        err_rdf = np.sqrt(np.var(blocked, axis=0, ddof=1) / blocked.shape[0])

        return avg_rdf - 1, err_rdf

    def dcf(self):

        blocked = block.block_data(self.raw_rdf, self.block_size)

        # Structure Factor via Fourier Transform
        sq_fft, q_fft = transforms.hr_to_sq(
            self.r_bins, self.rho, blocked - 1., self.r)
        self.q = q_fft

        # Correction of low q limit via direct measurement (Frankenstien's)
        spline = interp1d(self.q_, self.raw_sq, kind='cubic')
        mask = np.where(q_fft < np.max(self.q_))[0]
        q_dir = np.take(q_fft, mask)
        sq_dir = spline(q_dir)

        block_sq = block.block_data(sq_dir, self.block_size)
        sq_extend = np.pad(
            block_sq, ((0, 0), (0, len(q_fft) - len(q_dir))), 'constant')

        ind = 7
        switch = (1 + np.cos(np.pi * q_dir[:ind] / q_dir[ind])) * .5
        before = 0
        after = len(q_fft) - ind - before
        switch = np.pad(switch, (before, 0), 'constant', constant_values=(1))
        switch = np.pad(switch, (0, after), 'constant', constant_values=(0))
        sq_sw = switch * sq_extend + (1. - switch) * sq_fft

        # averages from different approaches
        avg_sq_dir = np.mean(sq_dir, axis=0)
        err_sq_dir = np.sqrt(np.var(sq_dir, axis=0, ddof=1) / sq_dir.shape[0])

        avg_sq_fft = np.mean(sq_fft, axis=0)
        err_sq_fft = np.sqrt(np.var(sq_fft, axis=0, ddof=1) / sq_fft.shape[0])

        avg_sq_sw = np.mean(sq_sw, axis=0)
        err_sq_sw = np.sqrt(np.var(sq_sw, axis=0, ddof=1) / sq_sw.shape[0])

        sdcf, f_r = transforms.sq_to_cr(self.r_bins, self.rho, sq_sw, self.q)
        avg_dcf = np.mean(sdcf, axis=0)
        err_dcf = np.sqrt(np.var(sdcf, axis=0, ddof=1) / sdcf.shape[0])

        # c(r) by fourier inversion for comparision
        blocked = block.block_data(self.raw_rdf, self.block_size)
        f_dcf = transforms.hr_to_cr(self.r_bins, self.rho, blocked - 1, self.r)
        avg_f_dcf = np.mean(f_dcf, axis=0)
        err_f_dcf = np.sqrt(np.var(f_dcf, axis=0, ddof=1) / f_dcf.shape[0])

        return avg_dcf, err_dcf, avg_sq_sw, err_sq_sw

    def ccf(self):
        cav = np.mean(self.raw_cav.T / self.mu, axis=1)
        err_cav = np.sqrt(np.var(self.raw_cav.T, axis=0,
                                 ddof=1) / self.raw_cav.T.shape[0])

        # todo: look at this in detail to ensure we are matching correctly
        dcav_extend = np.pad(cav, (0, len(self.r) - len(self.r_)), 'constant')
        rs_start = 1.0
        rs_end = self.r_[-1]
        rs = np.arange(0, rs_end - rs_start, self.r[0])
        before = np.floor(rs_start / self.r[0]).astype(int)
        after = np.ceil((self.r[-1] - rs_end) / self.r[0]).astype(int)
        switch = (1 + np.cos(np.pi * rs / rs[-1])) * .5
        switch = np.pad(switch, (before, 0), 'constant', constant_values=(1))
        switch = np.pad(switch, (0, after), 'constant', constant_values=(0))
        bulk_cav = np.exp(self.phi) * np.mean(self.raw_rdf, axis=0)

        avg_ccf = switch * dcav_extend + (1 - switch) * np.nan_to_num(bulk_cav)
        err_ccf = 0

        return avg_ccf, err_ccf


def process_set(path, number, bpart, cpart, temp, rho, directory):
    data = processing()
    try:
        data.load_data(path, number, bpart, cpart, temp, rho)
    except:
        return

    avg_tcf, err_tcf = data.rdf()
    avg_dcf, err_dcf, avg_sq, err_sq = data.dcf()
    avg_ccf, err_ccf = data.ccf()
    r = data.r
    q = data.q

    bridge = np.log(avg_ccf) + avg_dcf - avg_tcf

    grad_tcf = np.gradient(avg_tcf, r[0])
    grad_dcf = np.gradient(avg_dcf, r[0])

    output = np.column_stack((r, bridge,
                              avg_tcf, avg_dcf,
                              grad_tcf, grad_dcf,
                              err_dcf, err_tcf,
                              q, avg_sq))

    np.savetxt('{}processed_d{}_t{}_p{}.dat'.format(
        directory, rho, temp, number), output)


if __name__ == "__main__":
    filename = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(filename) as f:
        lines = f.read().splitlines()

    for line in lines:
        opt = parse.parse_str(line)
        # input_path = opt.output
        input_number = re.findall('\d+', opt.table)[-1]
        input_bpart = opt.bulk_part
        input_cpart = opt.cav_part
        input_density = opt.rho
        input_temp = opt.temp

        process_set(input_path, input_number, input_bpart, input_cpart,
                    input_temp, input_density, output_path)
