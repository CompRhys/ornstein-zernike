from __future__ import print_function
import sys
import re
import os
import numpy as np
from core import block, transforms, parse
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter
matplotlib.rcParams.update({'font.size': 12})


def plot_funcs( r, phi, 
                avg_tcf, err_tcf, 
                fd_gr, fd_gr_sg,  
                avg_dcf, err_dcf,
                avg_icf, err_icf, 
                avg_grad_icf, err_grad_icf,
                avg_dcf_dir, err_dcf_dir, 
                avg_dcf_fft, err_dcf_fft,
                avg_br, err_br):

    plt.plot(r, fd_gr)
    plt.plot(r, fd_gr_sg)
    plt.ylim((0,1e-3))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Plot g(r)

    axes[0, 0].plot(r, avg_tcf + 1)
    axes[0, 0].fill_between(r, avg_tcf + err_tcf + 1, avg_tcf - err_tcf + 1, alpha=0.3)

    axes[0, 0].plot((r[0],r[-1]), np.ones(2), '--', color="tab:blue")
    axes[0, 0].set_xlabel('r/$\sigma$')
    axes[0, 0].set_ylabel('$g(r)$')

    # Plot c(r)

    axes[0, 1].plot(r, avg_dcf_fft, label='$c_{fft}(r)$')
    axes[0, 1].fill_between(r, avg_dcf_fft + err_dcf_fft, avg_dcf_fft - err_dcf_fft, alpha=0.2)

    axes[0, 1].plot(r, avg_dcf, label='$c_{sw}(r)$')
    axes[0, 1].fill_between(r, avg_dcf + err_dcf, avg_dcf - err_dcf, alpha=0.2)

    axes[0, 1].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    axes[0, 1].set_xlabel('r/$\sigma$')
    axes[0, 1].set_ylabel('$c(r)$')
    axes[0, 1].legend()


    ## plot y(r)

    axes[0, 2].plot(r, avg_icf, label='$c_{sw}(r)$')
    axes[0, 2].fill_between(r, avg_icf + err_icf, avg_icf - err_icf, alpha=0.2)

    axes[0, 2].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    axes[0, 2].set_xlabel('r/$\sigma$')
    axes[0, 2].set_ylabel('$\gamma(r)$')
    axes[0, 2].legend()


    ## plot y'(r)

    axes[1, 2].plot(r, avg_grad_icf)
    axes[1, 2].fill_between(r, avg_grad_icf + err_grad_icf, 
                                avg_grad_icf - err_grad_icf, alpha=0.2)

    axes[1, 2].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    axes[1, 2].set_xlabel('r/$\sigma$')
    axes[1, 2].set_ylabel('$\gamma\'(r)$')
    axes[1, 2].legend()

    # Plot phi(r)

    axes[1, 0].plot(r, phi)
    axes[1, 0].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    axes[1, 0].set_ylim([-3, 8])
    axes[1, 0].set_xlabel('r/$\sigma$')
    axes[1, 0].set_ylabel('$\phi(r)$')

    # # Plot b(r)

    ind = len(r)-len(avg_br)

    axes[1, 1].plot(r[ind:], avg_br, label='$b_{sw}(r)$')
    axes[1, 1].fill_between(r[ind:], avg_br + err_br, 
                        avg_br - err_br, alpha=0.2)

    axes[1, 1].plot((r[0],r[-1]), np.zeros(2), '--', color="tab:blue")
    # axes[1, 2].set_xlim([0, 3.5])
    axes[1, 1].set_xlabel('r/$\sigma$')
    axes[1, 1].set_ylabel('$B(r)$')
    axes[1, 1].legend(loc=4)

    fig.tight_layout()

    return
    

def plot_sq_compare(q, switch, avg_sq, err_sq, avg_sq_fft, err_sq_fft,
                    avg_sq_switch, err_sq_switch, block_sq):
    # Plot s(q)
    matplotlib.rcParams.update({'font.size': 16})

    fig, axes = plt.subplots(2, figsize=(6,6), sharex=True, 
        gridspec_kw={'height_ratios':[8, 3]},)


    axes[0].plot(q, avg_sq, linewidth=1, marker='x', label='$S_{dir}(q)$')
    # axes[0].plot(q, block_sq.T, linewidth=1, marker='x', alpha=0.2)
    axes[0].fill_between(q, avg_sq + err_sq, avg_sq - err_sq, alpha=0.2)

    axes[0].plot(q, avg_sq_fft, color='tab:orange', linewidth=1, marker='o', mfc='none', label='$S_{fft}(q)$')
    axes[0].fill_between(q, avg_sq_fft + err_sq_fft, avg_sq_fft - err_sq_fft, alpha=0.2)

    axes[0].plot(q, avg_sq_switch, color='g', marker='+', linewidth=1, label='$S_{sw}(q)$')
    axes[0].fill_between(q, avg_sq_switch + err_sq_switch, avg_sq_switch - err_sq_switch, alpha=0.2)

    axes[0].plot(q, switch, color='r',linewidth=1, marker='*', label='W(q)')
    axes[0].set_xlim([0, 12.5])
    # axes[1, 0].set_ylim([-.5, 4.0])
    axes[0].set_ylabel('$S(q), W(q)$', labelpad=12)
    axes[0].legend(markerscale=1, fontsize=12, frameon=False)

    axes[1].plot(q, (- avg_sq + avg_sq_fft), linewidth=1, marker='x', label='$\Delta S(q)$')
    axes[1].fill_between(q, (- avg_sq + avg_sq_fft) + err_sq, (- avg_sq + avg_sq_fft) - err_sq, alpha=0.2)
    axes[1].plot((0,13), (0,0), 'k-.', linewidth=0.5)
    axes[1].set_xlabel('$q/\sigma^{-1}$')
    axes[1].set_ylabel('$\Delta S(q)$', labelpad=0)
    axes[1].set_xlim([0, 12.5])
    axes[1].set_yticks([-0.1, 0.2])
    axes[1].set_ylim([-0.15, 0.3])
    # axes[1].legend()
    fig.tight_layout()

    fig.subplots_adjust(hspace=0)

    return


if __name__ == "__main__":
    inputs = ' '.join(sys.argv[1:])

    opt = parse.parse_input(inputs)
    input_path = opt.output
    _, pot_type, pot_number = opt.table.split("_")
    pot_number = re.findall('\d+', pot_number)[-1]
    input_size = opt.box_size
    input_density = opt.rho
    input_temp = opt.temp

    print("potential {}_{}, density {}".format(pot_type, pot_number, input_density))
    
    n_part = int(input_density * (input_size**3.))

    rdf_path = '{}rdf_{}_{}_p{}_n{}_t{}.dat'.format(
        input_path, pot_type, pot_number, input_density, n_part, input_temp)
    sq_path = '{}sq_{}_{}_p{}_n{}_t{}.dat'.format(
        input_path, pot_type, pot_number, input_density, n_part, input_temp)
    phi_path = '{}phi_{}_{}.dat'.format(input_path, pot_type, pot_number)

    real, fourier = transforms.process_inputs(input_size, input_temp, input_density, 
                    "plot", rdf_path=rdf_path, sq_path=sq_path, phi_path=phi_path)

    plot_funcs(*real)

    plot_sq_compare(*fourier)

    plt.show()

