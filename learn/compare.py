from __future__ import print_function
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse


def main():
    path = os.path.expanduser('~') + '/closure/'
    test_path = path + 'data/train/test.dat'
    test_set = np.loadtxt(test_path)
    test_size = len(test_set)
    # np.random.shuffle(test_set)
    # test_size   = 1000

    X = test_set[:test_size, 2:4]
    X_nl = test_set[:test_size, 2:6]
    # phi         = test_set [:test_size,0]
    bridge = test_set[:test_size, 0]
    h = test_set[:test_size, 1]
    c = test_set[:test_size, 2]
    py = np.log(h - c + 1) + c - h

    mlp = load_model(path + 'learn/local.h5')
    bridge_mlp = mlp.predict(X)
    mlp_r2 = r2_score(bridge, bridge_mlp)

    print("Local MLP score = {:.5f}".strip().format(mlp_r2))

    mlp_nl = load_model(path + 'learn/non-local.h5')
    bridge_mlp_nl = mlp_nl.predict(X_nl)
    mlp_nl_r2 = r2_score(bridge, bridge_mlp_nl)

    print("Non-Local MLP score = {:.5f}".strip().format(mlp_nl_r2))

    hnc_r2 = r2_score(bridge, np.zeros_like(bridge))

    print("HNC score = {:.5f}".strip().format(hnc_r2))

    mse_hnc = mse(np.zeros_like(bridge), bridge)
    print(mse_hnc)

    mse_mlp = mse(bridge, bridge_mlp)
    print(mse_mlp)

    mse_mlp_nl = mse(bridge, bridge_mlp_nl)
    print(mse_mlp_nl)


# def plot_bridge(bridge, names):
    matplotlib.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.plot(np.zeros_like(bridge), bridge,  linestyle="None",
            marker="x", markersize=3, label='HNC')
    ax.plot(bridge_mlp, bridge,  linestyle="None",
            marker="x", markersize=3, label='Local MLP')
    ax.plot(bridge_mlp_nl, bridge, linestyle="None",
            marker="x", markersize=3, label='Non-Local MLP')    
    # ax.plot(py, bridge, linestyle="None",
    #         marker="x", markersize=3, label='PY')
    ax.plot([-2, 2], [-2, 2], linestyle="--", color='k')
    ax.legend(markerscale=3, fontsize=12)
    # ax.set_xlim([-1.05, 0.1])
    # ax.set_ylim([-1.05, 0.1])
    ax.set_xlabel('$B_{ML}(r)$')
    ax.set_ylabel('$B(r)$')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
