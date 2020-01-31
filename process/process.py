from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms, parse
from tqdm import tqdm

def process_input(input_path, pot_type, pot_number, box_size, temp, 
                    input_density, pass_path, fail_path):
    """
    
    cleaning: identify a method to discard glassy systems. Not critical unlikely 
    to see classes in single particle systems will be signifcantly more 
    important in investigations of two particle systems. possible approaches 
    involve looking at the intermediate scattering function (glasses won't 
    decay), estimating a non-gaussian parameter (caging effects), looking at 
    MSD for evidence of caged diffusion."""
    
    n_part = int(input_density * (box_size**3.))
    density = n_part / (box_size**3.)

    phi = np.loadtxt('{}phi_{}_{}.dat'.format(input_path, pot_type, pot_number))
    rdf = np.loadtxt('{}rdf_{}_{}_p{}_n{}_t{}.dat'.format(
        input_path, pot_type, pot_number, input_density, n_part, temp))
    sq = np.loadtxt('{}sq_{}_{}_p{}_n{}_t{}.dat'.format(
        input_path, pot_type, pot_number, input_density, n_part, temp))
    T  = np.loadtxt('{}temp_{}_{}_p{}_n{}_t{}.dat'.format(
        input_path, pot_type, pot_number, input_density, n_part, temp))[:,1]

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

    # block_size = 256
    block_tcf = block.block_data(tcf, block_size)
    block_sq = block.block_data(sq, block_size)

    # # optional 5 point function smoothing
    # block_tcf = transforms.smooth_function(block_tcf)
    # block_sq = transforms.smooth_function(block_sq)

    # h(r)
    avg_tcf = np.mean(block_tcf, axis=0)
    err_tcf = np.sqrt(np.var(block_tcf, axis=0, ddof=1) / block_tcf.shape[0])

    # h'(r)

    r_peak = r[np.argmax(avg_tcf)]
    grad_tcf = np.gradient(block_tcf * r_peak, r, axis=1)
    avg_grad_tcf = np.mean(grad_tcf ,axis=0)
    err_grad_tcf = np.sqrt(np.var(grad_tcf, axis=0, ddof=1) / block_tcf.shape[0])


    # Switching function w(q)
    # print(np.argmax(block_sq > 0.75*np.max(block_sq), axis=1))
    peak = np.median(np.argmax(block_sq > 0.75*np.max(block_sq), axis=1)).astype(int)
    after = len(q) - peak 
    switch = (1 + np.cbrt(np.cos(np.pi * q[:peak] / q[peak]))) / 2.
    switch = np.pad(switch, (0, after), 'constant', constant_values=(0))

    # s(q) from fft
    sq_fft, q_fft = transforms.hr_to_sq(r_bins, density, block_tcf, r)
    assert(np.all(np.abs(q-q_fft)<1e-10))

    # Corrected s(q) using switch

    sq_switch = switch * block_sq + (1. - switch) * sq_fft
    avg_sq_switch = np.mean(sq_switch, axis=0)
    err_sq_switch = np.sqrt(np.var(sq_switch, axis=0, ddof=1) / sq_switch.shape[0])

    ## Evaluate c(r)

    dcf_swtch, r_swtch = transforms.sq_to_cr(r_bins, density, sq_switch, q_fft)
    avg_dcf_swtch = np.mean(dcf_swtch, axis=0)
    err_dcf_swtch = np.sqrt(np.var(dcf_swtch, axis=0, ddof=1) / dcf_swtch.shape[0])

    ## Evaluate c'(r)

    grad_dcf_swtch = np.gradient(dcf_swtch * r_peak, r_swtch, axis=1)
    avg_grad_dcf_swtch = np.mean(grad_dcf_swtch ,axis=0)
    err_grad_dcf_swtch = np.sqrt(np.var(grad_dcf_swtch, axis=0, ddof=1) / dcf_swtch.shape[0])

    ## Evaluate B(r)
    bridge = np.log(avg_tcf + 1.) + phi - avg_tcf + avg_dcf_swtch


    output = np.column_stack((r, bridge, phi,
                                avg_tcf, err_tcf,
                                avg_grad_tcf, err_grad_tcf,
                                avg_dcf_swtch, err_dcf_swtch,
                                avg_grad_dcf_swtch, err_grad_dcf_swtch,
                                q, avg_sq_switch, err_sq_switch))

    block_T = block.block_data(T.reshape((-1,1)), block_size)
    err = np.std(block_T)
    res = np.abs(np.mean(block_T - temp))

    if res > err:
        np.savetxt('{}processed_{}_{}_p{}_n{}_t{}.dat'.format(
        fail_path, pot_type, pot_number, input_density, n_part, temp), output)
        print("fail temp, res {}, err {}".format(res, err))

    elif avg_sq_switch[0] > 1.0:
        np.savetxt('{}processed_{}_{}_p{}_n{}_t{}.dat'.format(
        fail_path, pot_type, pot_number, input_density, n_part, temp), output)
        print("fail two")

    elif np.max(avg_sq_switch) > 2.8:
        np.savetxt('{}processed_{}_{}_p{}_n{}_t{}.dat'.format(
        fail_path, pot_type, pot_number, input_density, n_part, temp), output)
        print("fail hansen")

    else:
        np.savetxt('{}processed_{}_{}_p{}_n{}_t{}.dat'.format(
        pass_path, pot_type, pot_number, input_density, n_part, temp), output)

    pass


if __name__ == "__main__":
    filename = sys.argv[1]
    pass_path = sys.argv[2]
    fail_path = sys.argv[3]

    with open(filename) as f:
        lines = f.read().splitlines()

    for line in tqdm(lines):
    # for line in lines:
        opt = parse.parse_input(line)
        raw_path = opt.output
        _, pot_type, pot_number = opt.table.split("_")
        pot_number = re.findall('\d+', pot_number)[-1]
        input_size = opt.box_size
        input_density = opt.rho
        input_temp = opt.temp

        process_input(raw_path, pot_type, pot_number, input_size,
            input_temp, input_density, pass_path, fail_path)
