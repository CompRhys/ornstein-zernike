from __future__ import print_function
import os
import sys
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse


def main():
    data_path = sys.argv[1]

    data_set = np.loadtxt(data_path, delimiter=",", skiprows=1)
    X = data_set[:, 2:6]
    y = data_set[:, 1]
    # n = data_set[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_local = X_test[:, 0:2]
    X_nl = X_test[:, 0:4]
    # phi         = test_set [:test_size,0]
    bridge = y_test

    mlp = load_model('learn/local.h5')
    bridge_mlp = mlp.predict(X_local)
    mlp_r2 = r2_score(bridge, bridge_mlp)

    print("Local MLP score = {:.5f}".strip().format(mlp_r2))

    mlp_nl = load_model('learn/non-local.h5')
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
    ax.set_xlim([-1.05, 0.1])
    ax.set_ylim([-1.05, 0.1])
    ax.set_xlabel('$B_{ML}(r)$')
    ax.set_ylabel('$B(r)$')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
