import numpy as np
from scipy.fftpack import dst, idst
# from scipy.interpolate import UnivariateSpline as Spline
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import savgol_filter
from scipy.integrate import simps
from core import block

def hr_to_cr(bins, rho, data, radius, error=None, axis=1):
    """
    This function takes h(r) and uses the OZ equation to find c(r) this is done via a 3D fourier transform
    that is detailed in LADO paper. The transform is the the DST of f(r)*r. The function is rearranged in
    fourier space to find c(k) and then the inverse transform is taken to get back to c(r).
    """
    # setup scales

    dk = np.pi / radius[-1]
    k = dk * np.arange(1, bins + 1, dtype=np.float)

    # Transform into fourier components
    FT = dst(data * radius[0:bins], type=1, axis=axis)
    normalisation = 2 * np.pi * radius[0] / k
    H_k = normalisation * FT

    # Rearrange to find direct correlation function

    C_k = H_k / (1 + rho * H_k)

    # # Transform back to real space
    iFT = idst(C_k * k, type=1)
    normalisation = k[-1] / (4 * np.pi**2 * radius[0:bins]) / (bins + 1)
    c_r = normalisation * iFT

    return c_r, radius


def hr_to_sq(bins, rho, data, radius, axis=1):
    """
    this function takes h(r) and takes the fourier transform to find s(k)
    """
    # setup scales

    dk = np.pi/radius[-1]
    k = dk * np.arange(1, bins + 1, dtype=np.float)

    # Transform into fourier components
    FT = dst(data * radius[0:bins], type=1, axis=axis)
    # radius[0] is dr as the bins are spaced equally.
    normalisation = 2 * np.pi * radius[0] / k
    H_k = normalisation * FT

    S_k = 1 + rho * H_k

    return S_k, k


def sq_to_hr(bins, rho, S_k, k, axis=1):
    """
    Takes the structure factor s(q) and computes the real space 
    total correlation function h(r)
    """
    # setup scales

    dr = np.pi / (k[0] * bins)
    radius = dr * np.arange(1, bins + 1, dtype=np.float)

    # Rearrange to find total correlation function from structure factor
    H_k = (S_k - 1.) / rho

    # # Transform back to real space
    iFT = idst(H_k * k[:bins], type=1, axis=axis)
    normalisation = bins * k[0] / (4 * np.pi**2 * radius) / (bins + 1)
    h_r = normalisation * iFT

    return h_r, radius


def sq_to_cr(bins, rho, S_k, k, axis=1):
    """
    Takes the structure factor s(q) and computes the direct correlation 
    function in real space c(r)
    """
    # setup scales

    dr = np.pi / (bins * k[0])
    radius = dr * np.arange(1, bins + 1, dtype=np.float)

    # Rearrange to find direct correlation function from structure factor
    # C_k = (S_k-1.)/(S_k) # 1.-(1./S_k) what is better
    C_k = (S_k - 1.) / (rho * S_k)

    # # Transform back to real space
    iFT = idst(k[:bins] * C_k, type=1, axis=axis)
    normalisation = bins * k[0] / (4 * np.pi**2 * radius) / (bins + 1)
    c_r = normalisation * iFT

    return c_r, radius


def sq_and_hr_to_cr(bins, rho, hr, r, S_k, k, axis=1):
    """
    Takes the structure factor s(q) and computes the direct correlation 
    function in real space c(r)
    """
    # setup scales

    dr = np.pi / (bins * k[0])
    radius = dr * np.arange(1, bins + 1, dtype=np.float)

    assert(np.all(np.abs(radius-r)<1e-12))

    iFT = idst(k[:bins] * np.square(S_k - 1.)/(rho * S_k), type=1, axis=axis) 

    cr = hr - iFT

    return cr


def smooth_function(f):
    """
    five point smoothing as detailed on page 204 of Computer Simulation of Liquids.
    """

    g = np.zeros_like(f)

    g[:, 0] = 1. / 70. * (69 * f[:, 0] + 4 * f[:, 1] -
                          6 * f[:, 2] + 4 * f[:, 3] - f[:, 4])
    g[:, 1] = 1. / 35. * (2 * f[:, 0] + 27 * f[:, 1] +
                          12 * f[:, 2] - 8 * f[:, 3] + 2 * f[:, 4])
    g[:, -2] = 1. / 35. * (2 * f[:, -1] + 27 * f[:, -2] +
                           12 * f[:, -4] - 8 * f[:, -4] + 2 * f[:, -5])
    g[:, -1] = 1. / 70. * (69 * f[:, -1] + 4 * f[:, -2] -
                           6 * f[:, -3] + 4 * f[:, -4] - f[:, -5])

    for i in np.arange(2, f.shape[1] - 2):
        g[:, i] = 1. / 35. * (-3 * f[:, i - 2] + 12 * f[:, i - 1] +
                              17 * f[:, i] + 12 * f[:, i + 1] - 3 * f[:, i + 2])

    return g


def quadratic_spline_roots(spl):
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(roots)

def spline_max(r, tcf):
    """
    we get gradients from taking a cubic spline and then calculating the values for the 
    gradients from the resulting quadratic spline obtained by taking derivatives of the
    function spline. In order to define the gradients in a consistent manner we define
    them wrt to a lengthscale imposed by the principle peak.
    """

    spl_tcf = Spline(r, tcf, k=3)
    cr_pts = quadratic_spline_roots(spl_tcf.derivative())
    cr_pts = np.append(cr_pts, (r[0], r[-1]))  # also check the endpoints of the interval
    cr_vals = spl_tcf(cr_pts)
    max_index = np.argmax(cr_vals)
    # print("Maximum value {} at {}".format(cr_vals[max_index], cr_pts[max_index]))
    # min_index = np.argmin(cr_vals)
    # print("Minimum value {} at {}".format(cr_vals[min_index], cr_pts[min_index]))
    return cr_pts[max_index]

def process_inputs(box_size, temp, input_density, output="process", 
                    **paths):
    """
    """
    if output == "invert":
        assert len(paths) == 2, "rdf_path and sq_path must be provided"
    elif output == "plot":
        assert len(paths) == 3, "rdf_path, sq_path and phi_path must be provided"
    elif output == "process":
        assert len(paths) == 4, "rdf_path, sq_path, phi_path and temp_path must be provided"
    else:
        raise ValueError("Unknown output given - direct/plot/process")

    n_part = int(input_density * (box_size**3.))
    density = n_part / (box_size**3.)

    rdf = np.loadtxt(paths.get('rdf_path'))
    sq = np.loadtxt(paths.get('sq_path'))

    r = rdf[0, :]
    r_bins = len(r)
    tcf = rdf[1:, :] - 1.

    q = sq[0, :]
    sq = sq[1:, :]

    # Find block size to remove correlations
    block_size_tcf = block.fp_block_length(tcf)
    block_size_sq = block.fp_block_length(sq)
    block_size = np.max((block_size_tcf, block_size_sq))
    # print("number of observations is {}, \nblock size is {}. \npercent {}%.".format(rdf.shape[0]-1, block_size, block_size/rdf.shape[0]*100))

    # block_size = 256
    block_tcf = block.block_data(tcf, block_size)
    block_sq = block.block_data(sq, block_size)

    ind = np.median(np.argmax(block_tcf + 1. > 0.1, axis=1)).astype(int)

    # TCF
    avg_tcf = np.mean(block_tcf, axis=0)
    err_tcf = np.sqrt(np.var(block_tcf, axis=0, ddof=1) / block_tcf.shape[0])

    r_peak = r[np.argmax(avg_tcf)]
    
    grad_tcf = np.gradient(block_tcf, r, axis=1)*r_peak
    avg_grad_tcf = np.mean(grad_tcf, axis=0)
    err_grad_tcf = np.sqrt(np.var(grad_tcf, axis=0, ddof=1) / block_tcf.shape[0])

    grad_tcf_sg = savgol_filter(block_tcf, window_length=11, polyorder=3, deriv=1, delta=r[1]-r[0], axis=1)*r_peak
    grad_tcf_sg[:,:ind] = grad_tcf[:,:ind]
    avg_grad_tcf_sg = np.mean(grad_tcf_sg, axis=0)
    err_grad_tcf_sg = np.sqrt(np.var(grad_tcf_sg, axis=0, ddof=1) / block_tcf.shape[0])

    # s(q)
    avg_sq = np.mean(block_sq, axis=0)
    err_sq = np.sqrt(np.var(block_sq, axis=0, ddof=1) / block_sq.shape[0])

    # s(q) from fft
    sq_fft, q_fft = hr_to_sq(r_bins, density, block_tcf, r)
    assert np.all(np.abs(q-q_fft)<1e-10), "The fft and sq wave-vectors do not match"

    avg_sq_fft = np.mean(sq_fft, axis=0)
    err_sq_fft = np.sqrt(np.var(sq_fft, axis=0, ddof=1) / sq_fft.shape[0])

    # Switching function w(q)
    # print(np.argmax(block_sq > 0.75*np.max(block_sq), axis=1))
    peak = np.median(np.argmax(block_sq.T > 0.75*np.max(block_sq, axis=1), axis=0)).astype(int)

    after = len(q_fft) - peak 
    switch = (1 + np.cbrt(np.cos(np.pi * q[:peak] / q[peak]))) / 2.
    switch = np.pad(switch, (0, after), 'constant', constant_values=(0))

    # Corrected s(q) using switch

    sq_switch = switch * block_sq + (1. - switch) * sq_fft
    avg_sq_switch = np.mean(sq_switch, axis=0)
    err_sq_switch = np.sqrt(np.var(sq_switch, axis=0, ddof=1) / sq_switch.shape[0])

    ## Evaluate c(r)

    # evaluate c(r) from corrected s(q)
    dcf_swtch, r_swtch = sq_to_cr(r_bins, density, sq_switch, q_fft)
    avg_dcf_swtch = np.mean(dcf_swtch, axis=0)
    err_dcf_swtch = np.sqrt(np.var(dcf_swtch, axis=0, ddof=1) / dcf_swtch.shape[0])

    grad_dcf_swtch_sg = savgol_filter(dcf_swtch, window_length=11, polyorder=3, deriv=1, delta=r[1]-r[0], axis=1)*r_peak
    avg_grad_dcf_swtch_sg = np.mean(grad_dcf_swtch_sg ,axis=0)
    err_grad_dcf_swtch_sg = np.sqrt(np.var(grad_dcf_swtch_sg, axis=0, ddof=1) / dcf_swtch.shape[0])

    # # c(r) by fourier inversion of just convolved term for comparision
    # dcf_both = transforms.sq_and_hr_to_cr(r_bins, input_density, block_tcf, r, block_sq, q)
    # avg_dcf_both = np.mean(dcf_both, axis=0)
    # err_dcf_both = np.sqrt(np.var(dcf_both, axis=0, ddof=1) / dcf_both.shape[0])

    # signs = np.where(np.sign(avg_tcf[:-1]) != np.sign(avg_tcf[1:]))[0] + 1
    # print(signs)

    # kbi_avg_1 = simps(r[:987]**2 * avg_tcf[:987], r[:987])
    # kbi_avg_2 = simps(r[:937]**2 * avg_tcf[:937], r[:937])
    # kbi_avg_3 = simps(r[:894]**2 * avg_tcf[:894], r[:894])

    # print(kbi_avg_1, kbi_avg_2, kbi_avg_3)

    if output == "plot":
        # evaluate c(r) from h(r)
        dcf_fft, r_fft = hr_to_cr(r_bins, density, block_tcf, r)
        avg_dcf_fft = np.mean(dcf_fft, axis=0)
        err_dcf_fft = np.sqrt(np.var(dcf_fft, axis=0, ddof=1) / dcf_fft.shape[0])

        # evaluate c(r) from s(q)
        dcf_dir, r_dir = sq_to_cr(r_bins, density, block_sq, q)
        avg_dcf_dir = np.mean(dcf_dir, axis=0)
        err_dcf_dir = np.sqrt(np.var(dcf_dir, axis=0, ddof=1) / dcf_dir.shape[0])

        # calculate non-smoothed gradient
        grad_dcf_swtch = np.gradient(dcf_swtch, r_swtch, axis=1)*r_peak
        avg_grad_dcf_swtch = np.mean(grad_dcf_swtch ,axis=0)
        err_grad_dcf_swtch = np.sqrt(np.var(grad_dcf_swtch, axis=0, ddof=1) / dcf_swtch.shape[0])

    ## Evaluate B(r)
    if output == "invert":
        return (r, avg_tcf, err_tcf, avg_grad_tcf_sg, err_grad_tcf_sg, 
                avg_dcf_swtch, err_dcf_swtch, avg_grad_dcf_swtch_sg, err_grad_dcf_swtch_sg)

    phi = np.loadtxt(paths.get('phi_path'))
    assert np.all(np.abs(r-phi[0,:])<1e-10), "the rdf and phi radii do not match" 
    phi = phi[1,:]

    br_swtch = np.log((block_tcf[:,ind:] + 1.)) + np.repeat(phi[ind:].reshape(-1,1), 
                    block_tcf.shape[0], axis=1).T- block_tcf[:,ind:] + dcf_swtch[:,ind:]
    avg_br_swtch = np.mean(br_swtch, axis=0)
    err_br_swtch = np.sqrt(np.var(br_swtch, axis=0, ddof=1) / br_swtch.shape[0])

    if output == "plot":
        return (r, phi, avg_tcf, err_tcf, avg_grad_tcf, err_grad_tcf, 
                avg_grad_tcf_sg, err_grad_tcf_sg, 
                avg_dcf_swtch, err_dcf_swtch, avg_grad_dcf_swtch, err_grad_dcf_swtch,
                avg_grad_dcf_swtch_sg, err_grad_dcf_swtch_sg,
                avg_dcf_dir, err_dcf_dir, avg_dcf_fft, err_dcf_fft,
                avg_br_swtch, err_br_swtch), \
                (q, switch, avg_sq, err_sq, avg_sq_fft, err_sq_fft,
                avg_sq_switch, err_sq_switch, block_sq)
    else:
        avg_br_swtch = np.pad(avg_br_swtch, (ind,0), "constant", constant_values=np.NaN)
        err_br_swtch = np.pad(err_br_swtch, (ind,0), "constant", constant_values=np.NaN)

        # Check if data satifies our cleaning heuristics
        T  = np.loadtxt(paths.get('temp_path'))[:,1]
        block_T = block.block_data(T.reshape((-1,1)), block_size)
        err = np.std(block_T)
        res = np.abs(np.mean(block_T - temp))

        if res > err:
            passed = False
        elif avg_sq_switch[0] > 1.0:
            passed = False
        elif np.max(avg_sq_switch) > 2.8:
            passed = False
        else:
            passed = True

        return passed, (r, phi, avg_tcf, err_tcf, avg_grad_tcf_sg, err_grad_tcf_sg, 
                avg_dcf_swtch, err_dcf_swtch, avg_grad_dcf_swtch_sg, err_grad_dcf_swtch_sg,
                avg_br_swtch, err_br_swtch)

