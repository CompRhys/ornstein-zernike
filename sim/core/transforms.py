import numpy as np
from scipy.fftpack import dst, idst

def hr_to_cr(bins, rho, data, radius, error=None, axis=1):
    """
    This function takes h(r) and uses the OZ equation to find c(r) this is done via a 3D fourier transform
    that is detailed in LADO paper. The transform is the the DST of f(r)*r. The function is rearranged in
    fourier space to find c(k) and then the inverse transform is taken to get back to c(r).
    This routine in addition to the functionality of the above routine takes an error in h(r) and 
    propagates it though to find the error in c(r). However as s(q) can be ~ zero the error blows up due
    to a 1/s(q)^2 term in the error calculatiom. Because of this propagating the error in the average appears
    to the order of the propagation to incorrect.
    """
    # setup scales

    dk = np.pi / radius[-1]
    k = dk * np.arange(1,bins+1, dtype=np.float)

    # Transform into fourier components
    FT = dst(data * radius[0:bins], type=1, axis=axis) 
    normalisation = 2 * np.pi * radius[0] / k
    H_k = normalisation*FT

    # Rearrange to find direct correlation function

    C_k = H_k/(1+rho*H_k) 

    # # Transform back to real space
    iFT = idst(C_k * k, type=1)
    normalisation = k[-1] / (4 * np.pi**2 * radius[0:bins]) / (bins + 1)
    c_r = normalisation * iFT

    if error is not None:    
        sigma_FT = dst(error * radius[0:bins], type=1, axis=1)
        normalisation = 2 * np.pi * delta_r / k
        sigma_H_K = normalisation*sigma_FT
        sigma_C_K = np.absolute(sigma_H_K/np.square(1+rho*H_k))
        sigma_iFT = idst(sigma_C_K * k, type=1)
        normalisation = k[-1] / (4 * np.pi**2 * radius[0:bins]) / (bins + 1)
        sigma_c_r = normalisation * sigma_iFT 
        return c_r, sigma_c_r
    else:
        return c_r



def hr_to_sq(bins, rho, data, radius,  axis=1):
    """
    this function takes h(r) and takes the fourier transform to find s(k)
    """
    # setup scales
    dk = np.pi / radius[-1]
    k = dk * np.arange(1,bins+1, dtype=np.float)

    # Transform into fourier components
    FT = dst(data * radius[0:bins], type=1, axis=axis) 
    normalisation = 2 * np.pi * radius[0] / k # radius[0] is dr as the bins are spaced equally.
    H_k = normalisation*FT

    S_k = 1+rho*H_k

    return S_k, k


def hr_to_sq_fixed(bins, rho, data, radius, box_l, axis=1):
    """
    this function takes h(r) and takes the fourier transform to find s(k)
    """
    # setup scales
    dk = 2 * np.pi / box_l
    # dk = np.pi / radius[-1]
    k = dk * np.arange(1,bins+1, dtype=np.float)

    # Transform into fourier components
    FT = dst(data * radius[0:bins], type=1, axis=axis) 
    normalisation = 2 * np.pi * radius[0] / k # radius[0] is dr as the bins are spaced equally.
    H_k = normalisation*FT

    S_k = 1+rho*H_k

    return S_k, k



def sq_to_hr(bins, rho, S_k, k, axis=1):
    """
    Takes the structure factor s(q) and computes the real space total correlation function h(r)
    """
    # setup scales

    dr = np.pi / (k[0] * bins)
    radius = dr * np.arange(1,bins+1, dtype=np.float)

    # Rearrange to find total correlation function from structure factor
    H_k = (S_k-1.)/rho

    # # Transform back to real space
    iFT = idst(H_k * k[:bins], type=1, axis=axis)
    normalisation = bins * k[0] / (4 * np.pi**2 * radius) / (bins + 1)
    h_r = normalisation * iFT

    return h_r, radius



def sq_to_cr(bins, rho, S_k, k, axis=1):
    """
    Takes the structure factor s(q) and computes the direct correlation function in real space c(r)
    """
    # setup scales

    dr = np.pi / (bins * k[0])
    radius = dr * np.arange(1,bins+1, dtype=np.float)

    # Rearrange to find direct correlation function from structure factor
    # C_k = (S_k-1.)/(S_k) # 1.-(1./S_k) what is better
    C_k = (S_k-1.)/(rho*S_k) 

    # # Transform back to real space
    iFT = idst(k[:bins] * C_k, type=1, axis=axis)
    normalisation = bins * k[0] / (4 * np.pi**2 * radius) / (bins + 1)
    c_r = normalisation * iFT

    return c_r, radius

