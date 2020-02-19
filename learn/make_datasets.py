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

            pot_type = labels.index(pot_type) * np.ones((test_case.shape[0]))
            pot_id = float(pot_id) * np.ones((test_case.shape[0]))
            density = float(density[1:]) * np.ones((test_case.shape[0]))

            output = np.column_stack((pot_type[start:],pot_id[start:],density[start:],test_case[start:,:]))

            if data is not None:
                data = np.vstack((data, output))
            else:
                data = output

    return data

def take_group(data, axis, group=[]):
    """
    Extract a sub-section from an array based on matching given criteria.
    """

    mask = np.argwhere(np.isin(data[:,axis], group)).ravel()
    test_set = np.take(data, mask, axis=0)
    train_set = np.delete(data, mask, axis=0)

    print("TRAIN: {} TEST: {}".format(np.unique(train_set[:,axis]), np.unique(test_set[:,axis])))
    return test_set, train_set


def main(directory, path):
    data = get_data(directory)

    names = ["pot_type", "pot_id", "density", "r", "phi", "tcf", "err_tcf", "grad_tcf", "err_grad_tcf",
            "dcf", "err_dcf", "grad_dcf", "err_grad_dcf", "br_swtch", "err_br_swtch"]

    assert len(names) == data.shape[1]

    names = ','.join(names)

    np.savetxt(path+'whole.csv', data, delimiter=',', header=names)

    hard_potentials, soft_potentials = take_group(data, 0, hard+core)

    print(hard_potentials.shape, soft_potentials.shape)

    hard_train, hard_test = train_test_split( hard_potentials, 
        test_size=0.2,  random_state=123)
    soft_train, soft_test = train_test_split(soft_potentials, 
        test_size=0.2,  random_state=123)

    whole_train = np.vstack((hard_train, soft_train))
    whole_test = np.vstack((hard_test, soft_test))

    np.savetxt(path+'random-train.csv', whole_train, delimiter=',', header=names)
    np.savetxt(path+'random-test.csv', whole_test, delimiter=',', header=names)

    train_size = min(hard_train.shape[0], soft_train.shape[0])
    test_size = min(hard_test.shape[0], soft_test.shape[0])

    np.savetxt(path+'hard-train.csv', hard_train[:train_size,:], delimiter=',', header=names)
    np.savetxt(path+'hard-test.csv', hard_test[:test_size,:], delimiter=',', header=names)

    np.savetxt(path+'soft-train.csv', soft_train[:train_size,:], delimiter=',', header=names)
    np.savetxt(path+'soft-test.csv', soft_test[:test_size,:], delimiter=',', header=names)
    
    density_test, density_train = take_group(data, 2, [0.8])

    np.savetxt(path+'density-train.csv', density_train, delimiter=',', header=names)
    np.savetxt(path+'density-test.csv', density_test, delimiter=',', header=names)


if __name__ == '__main__':
    directory = sys.argv[1]
    path = sys.argv[2]

    main(directory, path)
