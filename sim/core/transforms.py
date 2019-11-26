import numpy as np
from scipy.fftpack import dst, idst


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


def smooth_function(f,n):
    for i in range(n):
        f = smooth_func(f)
    return f


def smooth_func(f):
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
