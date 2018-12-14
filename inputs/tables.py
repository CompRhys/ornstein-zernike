'''
Script for generating the input tables for the different potentials used.
'''
import os
import numpy as np


def main():
    base_path = os.path.expanduser('~') 
    working_path = '/masters/closure/data/tables/'
    output_path = base_path + working_path

    r_min = 0.0
    r_max = 2.0
    samples = 4096
    rad = np.linspace(r_min, r_max, samples)

    # single minima
    lj(output_path, rad)
    morse(output_path, rad)

    # repulsive
    soft(output_path, rad)
    yukawa(output_path, rad)
    wca(output_path)

    # double minima
    dlvo(output_path, rad)
    exp_well(output_path, rad)

    # step potenstials
    step(output_path, rad)
    csw(output_path, rad)
    rssaw(output_path, rad)

    # soft potentials
    gaussian(output_path, rad)
    hat(output_path)
    hertzian(output_path)

    llano(output_path)

    pass


def lj(path, r):
    ptype = 'lj'
    energy = [0.65, 0.6, 0.55, 0.5]

    test_number = 0

    r6 = np.zeros_like(r)
    r6[1:] = np.power(1. / r[1:], 6.)
    r6[0] = 2 * r6[1] - r6[2]
    r12 = np.power(r6, 2.)

    for p2 in energy:

        potential = 4 * p2 * (r12 - r6)

        save_table(path, ptype, test_number, r, potential)

        test_number += 1


def soft(path, r):
    ptype = 'soft'
    energy = [1.0, 0.6]
    exponent = [4.0, 6.0, 8.0, 10.0]

    test_number = 0

    for p3 in energy:
        for p5 in exponent:

            potential = np.zeros_like(r)
            potential[1:] = p3 * (1. / r[1:])**p5
            potential[0] = 2 * potential[1] - potential[2]

            save_table(path, ptype, test_number, r, potential)

            test_number += 1


def morse(path, r):
    ptype = 'morse'
    energy = [0.65, 0.6, 0.55]
    minimum = [1.0]
    alpha = [5.0, 9.0]

    test_number = 0

    for p2 in energy:
        for p3 in minimum:
            for p4 in alpha:

                exp1 = np.exp(-2. * p4 * (r - p3))
                exp2 = np.exp(-p4 * (r - p3))

                potential = p2 * (exp1 - 2. * exp2)

                save_table(path, ptype, test_number, r, potential)

                test_number += 1




def yukawa(path, r):
    ptype = 'yukawa'
    alpha = [2., 4., 6.]
    kappa = [2.5, 3.5]
    delta = 0.0

    test_number = 0
    potential = np.zeros_like(r)

    for p2 in alpha:
        for p3 in kappa:

            potential[1:] = p2 * np.exp(-p3 * (r[1:] - delta)) / r[1:]
            potential[0] = 2. * potential[1] - potential[2]

            save_table(path, ptype, test_number, r, potential)

            test_number += 1

def wca(path):
    ptype = 'wca'
    energy = [1.0, 0.6]
    powers = [[50, 49], [35, 22], [20, 14], [12, 6]]

    test_number = 0
    r_min = 0.0
    samples = 4096

    for i in np.arange(len(powers)):

        alpha = powers[i][0]
        beta = powers[i][1]
        r_max = np.power(alpha / beta, 1 / (alpha - beta))
        r = np.linspace(r_min, r_max, samples)
        rs = np.zeros_like(r)
        rh = np.zeros_like(r)
        rs[1:] = np.power(1. / r[1:], beta)
        rs[0] = 2 * rs[1] - rs[2]
        rh[1:] = rs[1:] * np.power(1 / r[1:], alpha - beta)
        rh[0] = 2 * rh[1] - rh[2]
        prefactor = alpha * np.power(alpha / beta, beta)

        for p2 in energy:

            potential = prefactor * p2 * (rh - rs) + p2

            save_table(path, ptype, test_number, r, potential)

            test_number += 1


def step(path, r):
    ptype = 'step'
    diameter = [.7]
    exponent = 12.
    energy = [1.0]
    kappa = [1.0, 3.0, 5.0, 7.0]
    sigma = [1.0, 1.5, 2.0]
    cutoff = 0.001

    test_number = 0

    for p2 in energy:
        for p4 in sigma:
            for p5 in kappa:

                phi_step = p2 / (1. + np.exp(2 * p5 * (r - p4)))

                phi_hard = np.zeros_like(r)
                phi_hard[1:] = np.power((1. / r[1:]), exponent)
                phi_hard[0] = 2. * phi_hard[1] - phi_hard[2]

                potential = phi_step + phi_hard

                save_table(path, ptype, test_number, r, potential)

                test_number += 1


def csw(path, r):
    ptype = 'csw'
    kappa = [15.0, 5.0]
    offset_r = [1.2, 1.6]  # offset_r is scaling factor from sigma == 1
    energy_r = [1.]
    energy_a = [2.0]
    delta_a = [0.1, 0.2]  # delta_a is scaling factor from sigma
    offset_a = [2.]  # offset_a is scaling factor from sigma

    test_number = 0

    for p2 in energy_a:
        for p4 in kappa:
            for p5 in offset_r:
                for p6 in energy_r:
                    for p7 in delta_a:
                        for p8 in offset_a:

                            phi_hard = np.zeros_like(r)
                            phi_hard[1:] = np.power((1. / r[1:]), 12)
                            phi_hard[0] = 2. * phi_hard[1] - phi_hard[2]

                            phi_step = p2 / (1. + np.exp(p4 * (r - p5)))

                            phi_gauss = p6 * \
                                np.exp(-0.5 * np.square((r - p8) / (p7)))

                            potential = phi_step + phi_hard - phi_gauss

                            save_table(path, ptype, test_number, r, potential)

                            test_number += 1


def rssaw(path, r):
    ptype = 'rssaw'
    sigma1 = [0.8, 1.15, 1.5]
    sigma2 = [1.0, 1.35]
    lamda1 = [0.5]
    lamda2 = [0.3]

    test_number = 0

    for p2 in lamda1:
        for p4 in lamda2:
            for p5 in sigma1:
                for p6 in sigma2:

                    phi_hard = np.zeros_like(r)
                    phi_hard[1:] = np.power((1. / r[1:]), 14)
                    phi_hard[0] = 2. * phi_hard[1] - phi_hard[2]

                    phi_step = p2 * np.tanh(10 * (r - p5))

                    phi_well = p4 * np.tanh(10 * (r - p6))

                    potential = phi_step + phi_hard - phi_well
                    save_table(path, ptype, test_number, r, potential)

                    test_number += 1


def dlvo(path, r):
    ptype = 'dlvo'
    energy = [0.185, 0.2, 0.215, 0.23, 0.245]

    test_number = 0

    r4 = np.zeros_like(r)
    r4[1:] = np.power((1. / r[1:]), 4.)
    r4[0] = 2 * r4[1] - r4[2]
    r8 = np.square(r4)
    r12 = np.power(r4, 3.)

    for p3 in energy:

        potential = p3 * r12 - r8 + r4

        save_table(path, ptype, test_number, r, potential)

        test_number += 1


def exp_well(path, r):
    ptype = 'exp-well'
    energy = [3.0, 5.0, 7.0]  # 0.5 - 11.0
    kappa = [10., 20., 30.]  # 0.5 - 30.0
    # kappa   = np.arange(0.5, 30.0, 0.5).tolist()
    shift = 0.3

    test_number = 0

    r4 = np.power((1. / (r + shift)), 4.)
    r4[0] = 2 * r4[1] - r4[2]
    r8 = np.square(r4)
    r12 = np.power(r4, 3.)

    for p3 in energy:
        for p5 in kappa:

            rexp = np.exp(-p5 * np.power((r - 1. + shift), 4)) / (r + shift)
            rexp[0] = 2 * rexp[1] - rexp[2]

            potential = p3 * (r12 - r8) + rexp

            save_table(path, ptype, test_number, r, potential)

            test_number += 1


def gaussian(path, r):
    ptype = 'gaussian'
    energy = [4.0, 6.0, 8.0, 10.0]
    inv_sigma = [1.0, 1.5, 2.0]

    test_number = 0

    for p3 in energy:
        for p2 in inv_sigma:

            potential = p3 * np.exp(-.5 * np.power((r * p2), 2))
            save_table(path, ptype, test_number, r, potential)

            test_number += 1


def hat(path):
    ptype = 'hat'
    force = [4.0, 6.0, 8.0, 10.0]
    cutoff = [2.0, 3.0]  # don't make sigma too small

    test_number = 0
    r_min = 0.0
    samples = 4096

    for p3 in force:
        for r_max in cutoff:

            r = np.linspace(r_min, r_max, samples)

            potential = p3 * (r - r_max) * ((r + r_max) / (2 * r_max) - 1.)

            save_table(path, ptype, test_number, r, potential)

            test_number += 1


def hertzian(path):
    ptype = 'hertzian'
    energy = [4.0, 6.0, 8.0, 10.0]
    cutoff = [2.0, 3.0]  # don't make sigma too small

    test_number = 0
    r_min = 0.0
    samples = 4096

    for p3 in energy:
        for r_max in cutoff:

            r = np.linspace(r_min, r_max, samples)

            potential = p3 * np.power((1. - r / r_max), 2.5)

            save_table(path, ptype, test_number, r, potential)

            test_number += 1


def llano(path):
    ptype = 'llano'
    energy = [2./3.]

    test_number = 0

    r_min       = 0.0
    r_max       = 2.5
    samples     = 4096
    r           = np.linspace(r_min,r_max,samples)

    r6 = np.zeros_like(r)
    r6[1:] = np.power(1. / r[1:], 6.)
    r6[0] = 2 * r6[1] - r6[2]
    r12 = np.power(r6, 2.)

    for p2 in energy:

        potential = 4 * p2 * (r12 - r6)

        save_table(path, ptype, test_number, r, potential)

        test_number += 1

def make_output(phi, r):
    force = np.gradient(phi, r[1])
    phi = phi + force[-1] * r[-1]
    phi = phi - phi[-1]  # shift phi
    force = force - force[-1]

    output = np.vstack((r, phi, force))

    return output


def save_table(path, ptype, number, radius, field):

    labels = ['lj', 'morse',                    # single minima
              'soft', 'yukawa', 'wca',          # repulsive
              'dlvo', 'exp-well',               # double minima
              'step', 'csw', 'rssaw',           # step potentials
              'gaussian', 'hat', 'hertzian',    # soft
              'llano']    

    output = make_output(field, radius)

    ref = str(labels.index(ptype) + 1)
    index = format(number, "02")
    np.savetxt('{}input_{}{}.dat'.format(path, ref, index), output)

if __name__ == '__main__':
    main()
