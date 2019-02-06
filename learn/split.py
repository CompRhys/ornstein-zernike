import os
import re
import sys
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


def join(directory):

    X = np.empty((0, 6))
    y = np.empty((0,))

    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            path_to_file = os.path.abspath(os.path.join(dirpath, file))
            number = re.findall('\d+', file)[-1]
            pot_type = np.floor(float(number)/100).astype(int)
            start = 0
            end = -1
            # if pot_type == 9:
            #     pass
            # else:
            test_case = np.loadtxt(path_to_file)
            form =  pot_type * np.ones((test_case.shape[0],1))
            output = np.column_stack((test_case[start:end,2:7],form[start:end]))
            X = np.vstack((X, output))
            y = np.concatenate((y, test_case[start:end,1]))

    return X, y

def main(directory, X_in, y_in):

    X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, 
        test_size=0.2,  random_state=43)

    train_set       = np.column_stack((y_train, X_train))
    test_set        = np.column_stack((y_test, X_test))

    np.savetxt(directory+'train.dat', train_set)
    np.savetxt(directory+'test.dat' , test_set)


if __name__ == '__main__':
    directory = sys.argv[1]
    outpath = sys.argv[2]

    a, b = join(directory)
    main(outpath, a, b)