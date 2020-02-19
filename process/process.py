from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import transforms, parse
from tqdm import tqdm

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

        n_part = int(input_density * (input_size**3.))

        temp_path = '{}temp_{}_{}_p{}_n{}_t{}.dat'.format(
            raw_path, pot_type, pot_number, input_density, n_part, input_temp)
        rdf_path = '{}rdf_{}_{}_p{}_n{}_t{}.dat'.format(
            raw_path, pot_type, pot_number, input_density, n_part, input_temp)
        sq_path = '{}sq_{}_{}_p{}_n{}_t{}.dat'.format(
            raw_path, pot_type, pot_number, input_density, n_part, input_temp)
        phi_path = '{}phi_{}_{}.dat'.format(raw_path, pot_type, pot_number)

        passed, data = transforms.process_inputs(input_size, input_temp, input_density,
                                                "process", rdf_path=rdf_path, sq_path=sq_path, 
                                                phi_path=phi_path, temp_path=temp_path)

        output = np.vstack(data).T
 
        names = ["r", "phi", "tcf", "err_tcf", "grad_tcf", "err_grad_tcf",
            "dcf", "err_dcf", "grad_dcf", "err_grad_dcf", "br_swtch", "err_br_swtch"]

        assert len(names) == output.shape[1]

        names = ','.join(names)

        if passed:
            np.savetxt('{}processed_{}_{}_p{}_n{}_t{}.csv'.format(
            pass_path, pot_type, pot_number, input_density, n_part, input_temp),
                output, delimiter=',', header=names)
        else:
            np.savetxt('{}processed_{}_{}_p{}_n{}_t{}.csv'.format(
            fail_path, pot_type, pot_number, input_density, n_part, input_temp), 
                output, delimiter=',', header=names)




        