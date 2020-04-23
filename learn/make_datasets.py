import os
import re
import sys
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
# from sklearn.preprocessing import StandardScaler

labels = ['lj', 'morse',                    # single minima
            'soft', 'yukawa', 'wca',          # repulsive
            'dlvo', 'exp-well',               # double minima
            'step', 'csw', 'rssaw',           # step potentials
            'gaussian', 'hat', 'hertzian',    # soft
            'llano']

hard = [0, 1, 4]
soft = [2, 3]
core = [5, 6, 7, 8, 9]
overlap = [10, 11, 12]

def get_data(directory):
    """
    """
    data = None

    print("Collecting together data\n")

    for dirpath, _, filenames in os.walk(directory):
        for file in tqdm(filenames):
            path_to_file = os.path.abspath(os.path.join(dirpath, file))
            _, pot_type, pot_id, density, _, _ = file.split("_")

            test_case = np.loadtxt(path_to_file, delimiter=',')

            start = test_case.shape[0] - np.argmax(np.flip(np.isnan(test_case[:,-1])))
            start_fg = test_case.shape[0] - np.argmax(np.flip(np.isnan(test_case[:,-1])))

            test_case[:,-3] *= np.sqrt(int(float(density[1:]) * (20.**3.)))

            pot_type = labels.index(pot_type) * np.ones((test_case.shape[0]))
            pot_id = float(pot_id) * np.ones((test_case.shape[0]))
            density = float(density[1:]) * np.ones((test_case.shape[0]))

            output = np.column_stack((pot_type[start:],pot_id[start:],density[start:],test_case[start:,:]))

            if data is not None:
                data = np.vstack((data, output))
            else:
                data = output

    return data


def main(directory, path):
    """
    """
    data = get_data(directory)

    names = ["pot_type", "pot_id", "density", "r", "phi", "avg_tcf", "err_tcf", "avg_dcf", "err_dcf", "avg_icf", "err_icf", 
                "avg_grad_icf", "err_grad_icf", "fd_gr", "avg_br", "err_br"]

    assert len(names) == data.shape[1]

    names = ','.join(names)

    np.savetxt(path+'whole.csv', data, delimiter=',', header=names)


if __name__ == '__main__':
    directory = sys.argv[1]
    path = sys.argv[2]

    main(directory, path)
