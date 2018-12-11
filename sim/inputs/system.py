import os
import re
import sys
import itertools


def file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def main(inpath):

    # tables = list(file_paths(inpath))
    # index = [None] * len(tables)
    # for i in range(len(tables)):
    #     index[i] = re.findall('\d+', tables[i])[0]
    # print(index)

    tables = file_paths(inpath)

    bulk_part = [1024]
    cav_part = [512]

    rho = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    temp = [1.]

    dt = [0.005]
    dr = [0.02]
    r_cav = [1.2]

    burn_steps = [1024]
    timesteps = [16]

    burn_iter_max = [16]
    bulk_iter = [1024]    # choose 2**n for fp method
    cav_iter = [10]
    # mu_iter = [4096]

    path = os.path.expanduser('~')
    output = [path + '/masters/closure/data/raw/']

    comb = itertools.product(tables, bulk_part, cav_part, rho, temp, dt, dr, r_cav,
                             burn_steps, timesteps, burn_iter_max, bulk_iter, cav_iter, output)

    with open('inputs.txt', 'w') as f:
        for li in comb:
            f.write(('--table {} --bulk_part {} --cav_part {} --rho {} --temp {} '
                     '--dt {} --dr {} --r_cav {} --burn_steps {} --timesteps {} '
                     '--burn_iter_max {} --bulk_iter {} --cav_iter {} --output {}\n').format(*li))

if __name__ == '__main__':
    inpath = sys.argv[1]
    main(inpath)
