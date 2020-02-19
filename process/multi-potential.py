from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms, parse
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.models import load_model

from scipy.signal import savgol_filter

matplotlib.rcParams.update({'font.size': 12})


def plot_funcs(r, avg_tcf, err_tcf, avg_grad_tcf_sg, err_grad_tcf_sg, 
                avg_dcf_swtch, err_dcf_swtch, avg_grad_dcf_swtch_sg, err_grad_dcf_swtch_sg):

    X_down = np.vstack((avg_tcf, avg_dcf_swtch, avg_grad_tcf_sg - avg_grad_dcf_swtch_sg)).T

    # # ind = np.argmax(avg_tcf + 1. > 0.1).astype(int)
    # ind_max = np.argmax(avg_grad_tcf_sg).astype(int)
    # ind_min = np.argmin(avg_grad_tcf_sg).astype(int)

    # # ind = (ind_max + np.where(np.sign(avg_grad_tcf_sg[ind_max:ind_min-1]) != np.sign(avg_grad_tcf_sg[ind_max+1:ind_min]))[0] + 1)[0]

    # # ind = ind_max
    ind = 0

    # tcf_ind = np.argmax(avg_tcf + 1. > 0.001).astype(int)

    # phi_ibi = np.zeros_like(avg_tcf)
    # phi_ibi[tcf_ind:] = -np.log(avg_tcf[tcf_ind:] + 1.)
    # phi_ibi[:tcf_ind] = np.arange(tcf_ind)*()
    # print(phi_ibi)

    phi_ibi = -np.log(avg_tcf + 1.)

    phi_id = np.min(np.argwhere(np.isfinite(phi_ibi)))

    phi_ibi[:phi_id] = phi_ibi[phi_id] + np.arange(1,phi_id+1)[::-1]*(phi_ibi[phi_id]-phi_ibi[phi_id+1])


    # Get HNC potential
    phi_hnc = avg_tcf - avg_dcf_swtch + phi_ibi

    # Get Local Closure Potential
    br_la = np.zeros_like(avg_tcf)
    local = load_model('learn/models/local-scaled-small-batch.h5', compile=False)
    br_la[ind:] = local.predict(X_down[ind:,0:2]).ravel()
    # br_la[ind:] = savgol_filter(local.predict(X_down[ind:,0:2]).ravel(), window_length=51, polyorder=3, delta=r[1]-r[0])

    phi_la  =  phi_hnc + br_la
    # phi_la -= local.predict(np.zeros((1,2))).ravel()

    # Get Non-Local Closure Potential
    br_nla = np.zeros_like(avg_tcf)
    non_local = load_model('learn/models/non-local-scaled.h5', compile=False)
    br_nla[ind:] = non_local.predict(X_down[ind:,0:3]).ravel()
    # br_nla[ind:] = savgol_filter(non_local.predict(X_down[ind:,0:4]).ravel(), window_length=21, polyorder=3, delta=r[1]-r[0])

    phi_nla  =  phi_hnc + br_nla
    # phi_nla -= non_local.predict(np.zeros((1,3))).ravel()



    # phi_nla_sg = savgol_filter(phi_nla, window_length=21, polyorder=3, delta=r[1]-r[0])

    # print(bridge.shape, phi_ibi.shape, phi_hnc.shape, phi_mlp.shape)

    output_path = "data/test/down/"

    # save phi
    f_phi = os.path.join(output_path,'phi_{}'.format("lj_mix"))
    # f_phi = os.path.join(output_path,'phi_{}'.format("ss_wca"))

    f_la = -savgol_filter(phi_la, window_length=11, polyorder=3, deriv=1 , delta=r[1]-r[0])
    r_cut_ind = np.argmin(r<3)

    # if os.path.isfile(f_phi):
    #     pass
    # else:
    phi_out = np.vstack((r[:r_cut_ind], phi_la[:r_cut_ind], f_la[:r_cut_ind]))
    np.savetxt(f_phi, phi_out)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Plot g(r)

    axes[0, 0].plot(r, avg_tcf + 1)
    axes[0, 0].fill_between(r, avg_tcf + err_tcf + 1, avg_tcf - err_tcf + 1, alpha=0.3)

    axes[0, 0].plot((r[0],r[-1]), np.ones(2), '--', color="tab:blue")
    axes[0, 0].set_xlabel('r/$\sigma$')
    axes[0, 0].set_ylabel('$g(r)$')

    axes[1, 0].plot(r, avg_grad_tcf_sg)
    axes[1, 0].fill_between(r, avg_grad_tcf_sg + err_grad_tcf_sg, avg_grad_tcf_sg - err_grad_tcf_sg, alpha=0.2)
    axes[1, 0].set_xlabel('r/$\sigma$')
    axes[1, 0].set_ylabel('$g\'(r)$')

    # Plot c(r)

    axes[0, 1].plot(r, avg_dcf_swtch, label='$c_{sw}(r)$')
    axes[0, 1].fill_between(r, avg_dcf_swtch + err_dcf_swtch, avg_dcf_swtch - err_dcf_swtch, alpha=0.2)

    axes[0, 1].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    axes[0, 1].set_xlabel('r/$\sigma$')
    axes[0, 1].set_ylabel('$c(r)$')
    axes[0, 1].legend()

    axes[1, 1].plot(r, avg_grad_dcf_swtch_sg)
    axes[1, 1].fill_between(r, avg_grad_dcf_swtch_sg + err_grad_dcf_swtch_sg, 
                            avg_grad_dcf_swtch_sg - err_grad_dcf_swtch_sg, alpha=0.2)
    axes[1, 1].set_xlabel('r/$\sigma$')
    axes[1, 1].set_ylabel('$c\'(r)$')

    axes[0, 2].plot(r, np.zeros_like(r), label="HNC")
    axes[0, 2].plot(r, br_la, label="LA")
    axes[0, 2].plot(r, br_nla, label="NLA")
    axes[0, 2].legend()

    # phi=np.loadtxt("data/raw/phi_csw_05.dat")[1,:].ravel()

    axes[1, 2].plot(r,phi_ibi, label="IBI")
    axes[1, 2].plot(r,phi_hnc, label="HNC")
    axes[1, 2].plot(r,phi_la ,  label="LA")
    axes[1, 2].plot(r,phi_nla,  label="NLA")
    # axes[1, 2].plot(r,phi_nla_sg,  label="NLA-SG")
    # axes[1, 2].plot(r,phi,  label="True")
    axes[1, 2].set_ylim((-2,5))
    axes[1, 2].legend()

    fig.tight_layout()

    plt.show()


    return
    



if __name__ == "__main__":

    input_size = 20.
    input_density = 0.8
    input_temp = 1.0

    rdf_path = sys.argv[1]
    sq_path = sys.argv[2]
    
    real = transforms.process_inputs(input_size, input_temp, input_density, 
                    "invert", rdf_path=rdf_path, sq_path=sq_path)

    plot_funcs(*real)

    plt.show()

