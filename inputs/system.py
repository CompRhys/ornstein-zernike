import os
import re
import sys
import itertools


def file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def main(inpath, output):
    tables = file_paths(inpath)

    rho = [0.4, 0.5, 0.6, 0.7, 0.8]
    temp = [1.]

    dt = [0.005]
    dr = [0.02]

    box_size = [20]

    bulk_steps = [16]
    bulk_iter = [8192]

    burn_steps = [2048]
    burn_iter_max = [16]

    comb = itertools.product(tables, rho, temp, dt, dr,
                             box_size, bulk_steps, bulk_iter,
                             burn_steps, burn_iter_max)

    outfilename = 'inputs.txt'
    with open(outfilename, 'w') as f:
        for li in comb:
            f.write(('--table {} --rho {} --temp {} --dt {} --dr {} '
                     '--box_size {} --bulk_steps {} --bulk_iter {} '
                     '--burn_steps {} --burn_iter_max {} --output {}\n').format(*li, output))

if __name__ == '__main__':
    inpath = sys.argv[1]
    output = sys.argv[2]

    main(inpath, output)
