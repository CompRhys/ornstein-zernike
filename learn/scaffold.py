import os
import re
import sys
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

hard = [1, 2, 5]
soft = [3, 4]
core = [6, 7, 8, 10]
overlap = [11, 12, 13]

mask = core + hard


def main(directory, outpath):

    X_train = np.empty((0, 5))
    y_train = np.empty((0,))
    X_test = np.empty((0, 5))
    y_test = np.empty((0,))

    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            path_to_file = os.path.abspath(os.path.join(dirpath, file))
            number = re.findall('\d+', file)[-1]
            pot_type = np.floor(float(number)/100).astype(int)

            if np.any(mask == pot_type):
                test_case = np.loadtxt(path_to_file)
                form =  pot_type * np.ones((test_case.shape[0],1))
                output = np.column_stack((test_case[:,2:6],form[:]))
                X_train = np.vstack((X_train, output))
                y_train = np.concatenate((y_train, test_case[:,1]))
            else:
                test_case = np.loadtxt(path_to_file)
                form =  pot_type * np.ones((test_case.shape[0],1))
                output = np.column_stack((test_case[:,2:6],form[:]))
                X_test = np.vstack((X_test, output))
                y_test = np.concatenate((y_test, test_case[:,1]))

    train_set       = np.column_stack((y_train, X_train))
    test_set        = np.column_stack((y_test, X_test))

    np.savetxt(directory+'train.dat', train_set)
    np.savetxt(directory+'test.dat' , test_set)


if __name__ == '__main__':
    directory = sys.argv[1]
    outpath = sys.argv[2]

    main(directory, outpath)