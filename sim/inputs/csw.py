import os
import numpy as np

def main():
    # Interaction parameters (Soft Sphere)
    kappa = [15.0, 5.0]
    offset_r = [1.2, 1.6]  # offset_r is scaling factor from sigma == 1
    energy_r = [1.]
    energy_a = [2.0]
    delta_a = [0.1, 0.2]  # delta_a is scaling factor from sigma
    offset_a = [2.]  # offset_a is scaling factor from sigma

    test_number = 0
    r_min = 0.0
    r_max = 3.0
    samples = 4096
    r = np.linspace(r_min, r_max, samples)


    for p2 in energy_a:
        for p4 in kappa:
            for p5 in offset_r:
                for p6 in energy_r:
                    for p7 in delta_a:
                        for p8 in offset_a:

                            phi_hard = np.zeros(samples)
                            phi_hard[1:] = np.power((1. / r[1:]), 12)
                            phi_hard[0] = 2. * phi_hard[1] - phi_hard[2]

                            phi_step = p2 / (1. + np.exp(p4 * (r - p5)))

                            phi_gauss = p6 * \
                                np.exp(-0.5 * np.square((r - p8) / (p7)))

                            potential = phi_step + phi_hard - phi_gauss
                            force = np.gradient(potential, r[1])
                            potential = potential + force[-1] * r_max
                            potential = potential - \
                                potential[-1]  # shift potential
                            force = force - force[-1]

                            tables = np.vstack((r, potential, force))
                            table_path = os.path.expanduser(
                                '~') + '/masters/closure/data/tables/'
                            np.savetxt(table_path + 'input_9' +
                                       format(test_number, "03") + '.dat', tables)

                            test_number += 1



if __name__ == '__main__':
    main()