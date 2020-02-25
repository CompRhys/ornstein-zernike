import sys
import re
import os
import numpy as np
from core import block, transforms, parse
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.models import load_model

from scipy.signal import savgol_filter, butter

from scipy.optimize import curve_fit


matplotlib.rcParams.update({'font.size': 12})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def backstrapolate(r, phi):
    assert len(r.shape) == len(phi.shape), "dimensions are not correct"

    cav = r.shape[0] - phi.shape[0]

    phi_ex = np.zeros_like(r)

    def polynom(r, a, b, c):
        return a * np.exp(b*r) + c

    popt, pcov = curve_fit(polynom, r[cav:cav+20], phi[:20])

    phi_ex[cav:] = phi
    phi_ex[:cav] = polynom(r[:cav], *popt)

    return r, phi_ex


def plot_funcs(r, avg_tcf, err_tcf, avg_dcf, err_dcf, avg_grad_icf, err_grad_icf, fd_gr,):
    """
    plot some stuff
    """
    mask_var = fd_gr.shape[0]- np.sum(np.isfinite(fd_gr))
    mask_avg = np.argmax(avg_tcf + 1.> 1e-8)

    print(mask_avg, mask_var, r[mask_var])
    mask = np.max((mask_avg, mask_var))

    X_down = np.vstack((avg_tcf[mask:], avg_dcf[mask:], fd_gr[mask:], avg_grad_icf[mask:])).T

    # Get IBI potential
    phi_ibi = -np.log(avg_tcf[mask:] + 1.)

    # Get HNC potential
    phi_hnc = avg_tcf[mask:] - avg_dcf[mask:] + phi_ibi

    ## Get Non-Local Closure Potential
    non_local = load_model('learn/models/non-local-400.h5', compile=False)
    br_nla = non_local.predict(X_down[:,0:4]).ravel()
    phi_nla  =  phi_hnc + br_nla

    br_nla_s = savgol_filter(br_nla, window_length=21, polyorder=2, deriv=0, delta=r[1]-r[0])
    phi_nla_s  =  phi_hnc + br_nla_s

    r, phi_ibi = backstrapolate(r, phi_ibi)
    r, phi_hnc = backstrapolate(r, phi_hnc)
    r, phi_nla_s = backstrapolate(r, phi_nla_s)
    r, phi_nla = backstrapolate(r, phi_nla)

    f_ibi = -np.gradient(phi_ibi, r[1]-r[0])
    f_hnc = -np.gradient(phi_hnc, r[1]-r[0])
    f_nla = -np.gradient(phi_nla, r[1]-r[0])
    # f_nla_s = -np.gradient(phi_nla_s, r[1]-r[0])

    # f_ibi = -savgol_filter(phi_ibi, window_length=31, polyorder=2, deriv=1, delta=r[1]-r[0])
    # f_hnc = -savgol_filter(phi_hnc, window_length=31, polyorder=2, deriv=1, delta=r[1]-r[0])
    f_nla_s = -savgol_filter(phi_nla_s, window_length=7, polyorder=2, deriv=1, delta=r[1]-r[0])

    #Save the potentials
    output_path = "data/test/down/"
    r_cut_ind = np.argmin(r<3)

    # save phi
    mix_type = "lj-mix"
    file_ibi = os.path.join(output_path,'phi_ibi_{}.dat'.format(mix_type))
    file_hnc = os.path.join(output_path,'phi_hnc_{}.dat'.format(mix_type))
    file_nla = os.path.join(output_path,'phi_nla_{}.dat'.format(mix_type))
    file_nla_n = os.path.join(output_path,'phi_nla_n_{}.dat'.format(mix_type))

    out_ibi = np.vstack((r[:r_cut_ind], phi_ibi[:r_cut_ind], f_ibi[:r_cut_ind]))
    np.savetxt(file_ibi, out_ibi)

    out_hnc = np.vstack((r[:r_cut_ind], phi_hnc[:r_cut_ind], f_hnc[:r_cut_ind]))
    np.savetxt(file_hnc, out_hnc)

    out_nla = np.vstack((r[:r_cut_ind], phi_nla_s[:r_cut_ind], f_nla_s[:r_cut_ind]))
    np.savetxt(file_nla, out_nla)

    out_nla_n = np.vstack((r[:r_cut_ind], phi_nla[:r_cut_ind], f_nla[:r_cut_ind]))
    np.savetxt(file_nla_n, out_nla_n)

    # plot

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Plot g(r)

    axes[0, 0].plot(r, avg_tcf + 1)
    axes[0, 0].fill_between(r, avg_tcf + err_tcf + 1, avg_tcf - err_tcf + 1, alpha=0.3)

    axes[0, 0].plot((r[0],r[-1]), np.ones(2), '--', color="tab:blue")
    axes[0, 0].set_xlabel('r/$\sigma$')
    axes[0, 0].set_ylabel('$g(r)$')

    # Plot c(r)

    axes[0, 1].plot(r, avg_dcf, label='$c_{sw}(r)$')
    axes[0, 1].fill_between(r, avg_dcf + err_dcf, avg_dcf - err_dcf, alpha=0.2)

    axes[0, 1].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    axes[0, 1].set_xlabel('r/$\sigma$')
    axes[0, 1].set_ylabel('$c(r)$')
    axes[0, 1].legend()

    # plot y'(r)

    axes[1, 1].plot(r, avg_grad_icf)
    axes[1, 1].fill_between(r, avg_grad_icf + err_grad_icf, avg_grad_icf - err_grad_icf, alpha=0.2)
    axes[1, 1].set_xlabel('r/$\sigma$')
    axes[1, 1].set_ylabel('$\gamma\'(r)$')

    # plot b(r)

    axes[0, 2].plot(r, np.zeros_like(r), label="HNC")
    axes[0, 2].plot(r[mask:], br_nla, label="NLA")
    axes[0, 2].plot(r[mask:], br_nla_s, label="NLA-SG")
    axes[0, 2].set_xlabel('r/$\sigma$')
    axes[0, 2].set_ylabel('$b(r)$')
    axes[0, 2].legend()

    # Phi(r)

    axes[1, 2].plot(r,phi_ibi, label="IBI")
    axes[1, 2].plot(r,phi_hnc, label="HNC")
    axes[1, 2].plot(r,phi_nla,  label="NLA", alpha=0.3)
    axes[1, 2].plot(r,phi_nla_s, label="NLA-SG")
    axes[1, 2].set_xlim((0,3))
    axes[1, 2].set_ylim((-2,6))
    axes[1, 2].set_xlabel('r/$\sigma$')
    axes[1, 2].set_ylabel('$\phi(r)$')
    axes[1, 2].legend()

    ## f(r)

    axes[1, 0].plot(r,f_ibi, label="IBI")
    axes[1, 0].plot(r,f_hnc, label="HNC")
    axes[1, 0].plot(r,f_nla, label="NLA", alpha=0.3)
    axes[1, 0].plot(r,f_nla_s, label="NLA-SG")
    # axes[1, 0].plot(r,f_nla_sg, '--', label="NLA-SG")
    axes[1, 0].set_xlim((0,3))
    axes[1, 0].set_ylim((-20,50))
    axes[1, 0].set_xlabel('r/$\sigma$')
    axes[1, 0].set_ylabel('$f(r)$')
    axes[1, 0].legend()

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

