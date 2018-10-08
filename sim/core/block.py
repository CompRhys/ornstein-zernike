"""
This routine is adapted from :
https://github.com/manoharan-lab/flyvbjerg-std-err

flyvbjerg_petersen_std_err is free software avaliable under the terms of
the GNU General Public License as published by the Free Software Foundation.

Copyright 2014, Jerome Fung, Rebecca W. Perry, Thomas G. Dimiduk

original aurthors:

.. moduleauthor:: Jerome Fung <jerome.fung@gmail.com>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Tom Dimiduk <tom@dimiduk.net>

Reference: H. Flyvbjerg and H. G. Petersen, "Error estimates on correlated
data", J. Chem. Phys. 91, 461--466 (1989).

In this script the routines have been adapted to find the best block length
for removing correlation from the data as opposed to the standard deviation
itself. This is necessary as we are taking the assumption that the correlation
times of all our structural measures should be the same.

The code has also been adapted to handl 2D arrays
"""

import numpy as np
import warnings

def block_data(series, length):
    """
    This function takes the data an combines it into blocks. Only works with powers of 2.
    """
    return np.mean(series.reshape(-1, length, series.shape[-1]), 1)



def block_transformation(series):
    """
    Do a single step of fp block averaging.

    Parameters
    ----------
    series : ndarray
        Things we want to average: e.g. squared displacements to calculate the
        mean squared displacement of a Brownian particle.

    Returns
    -------
    blocked_series : ndarray
        an array of half the length of series with adjacent terms averaged

    Notes
    -----
    Flyvbjerg & Peterson 1989, equation 20

    """
    n_steps = series.shape[0]
    if n_steps & 1:
       n_steps = n_steps - 1
    indices = np.arange(0, n_steps, 2)
    blocked_series = (np.take(series, indices, axis = 0) + np.take(series, indices+1, axis = 0))/2.
    return blocked_series

def calculate_blocked_variances(series, npmin = 4):
    """
    Compute a series of blocks and variances.

    Parameters
    ----------
    series : ndarray
        the thing we want to average: e.g. squared
        displacements for a Brownian random walk.
    npmin : int
        cutoff number of points to stop blocking

    Returns
    -------
    output_var, var_stderr : ndarray
        The variance and stderr of the variance at each blocking level

    Notes
    -----
    Flyvbjerg & Peterson suggest continuing blocking down to 2 points, but the
    last few blocks are very noisy, so we default to cutting off before that.

    """
    n_steps = series.shape[0]

    def block_var(d, n):
        # see eq. 27 of FP paper
        return np.var(d, axis = 0)/(n-1)
    def stderr_var(n):
        # see eq. 27 of FP paper
        return np.sqrt(2./(n-1))

    output_var = np.array([block_var(series, n_steps)]) # initialize
    var_stderr = np.array([stderr_var(n_steps)])

    while n_steps > npmin:
        series = block_transformation(series)
        n_steps = series.shape[0]
        # TODO: precompute size of output_var and var_stderr from n_steps
        # rather than appending
        x = block_var(series, n_steps)
        y = stderr_var(n_steps)
        output_var = np.vstack((output_var, x))
        var_stderr = np.append(var_stderr, y)

    return output_var, var_stderr

def detect_fixed_point(fp_nvar, fp_sev, full_output = False):
    """
    Find whether the block averages decorrelate the data series to a fixed
    point.

    Parameters
    ----------
    fp_var: ndarray
        FP blocked variance
    fp_sev: ndarray
        FP standard error of the variance.

    Returns
    -------
    best_var : float
        best estimate of the variance
    best_length : int
        best estimate of the block length
    converged : bool
        did the series converge to a fixed point?
    bounds : (int, int) only if full_output is True
        range of fp_var averaged to compute best_var

    Notes
    -----
    Expects both fp_var and fp_sev will have been
    truncated to cut off points with an overly small n_p and
    correspondingly large standard error of the variance.

    """
    n_trans = fp_nvar.shape[0] # number of block transformations and index
    n_samples = fp_nvar.shape[1] # number of variables

    left_index = np.zeros(n_samples, dtype=int)
    right_index = np.zeros(n_samples, dtype=int)
    best_var = np.zeros(n_samples)
    best_length = 1
    converged = np.zeros(n_samples, dtype=bool)


    for i in np.arange(n_samples):
        fp_var = fp_nvar[:,i]
        # Detect left edge
        for j in np.arange(n_trans)-1:
            # ith point inside error bars of next point
            if np.abs(fp_var[j + 1] - fp_var[j]) < fp_var[j + 1] * fp_sev[j + 1]:
                left_index[i] = j
                break
        # Check right edge
        for k in np.arange(n_trans)[::-1]:
            if np.abs(fp_var[k] - fp_var[k - 1]) < fp_var[k - 1] * fp_sev[k - 1]:
                right_index[i] = k
                break

        # if search succeeds
        if (left_index[i] >= 0) and (right_index[i] >= 0) and (right_index[i] >= left_index[i]):
            best_index = np.average(np.arange(left_index[i],right_index[i] + 1), 
                weights = 1./fp_sev[left_index[i]:right_index[i] + 1]).astype(np.int)
            best_length = max(best_length, np.power(2, best_index))
            converged[i] = True
        else:
            converged[i] = False

    if full_output is True:
        return best_length, converged, (left_index, right_index)
    else:
        return best_length, converged


def fp_block_length(data, conv_output = False):
    '''
    Compute standard error using Flyvbjerg-Petersen blocking.

    Computes the standard error on the mean of a possibly correlated timeseries
    of measurements.

    Parameters
    ----------
    data: ndarray
        data whose mean is to be calculated, and for which we need
        a standard error on the mean

    Returns
    -------
    stderr : float
        Standard error on the mean of data

    Notes
    -----

    Uses the technique described in H. Flyvbjerg and H. G. Petersen,
    "Error estimates on correlated data", J. Chem. Phys. 91, 461--466 (1989).
    section 3.

    '''
    data = np.atleast_1d(data)

    if len(data.shape) > 2:
        raise ValueError("invalid dimensions, input 2d or 1d array")

    block_trans_var, block_trans_sev = calculate_blocked_variances(data)
    len_block, conv = detect_fixed_point(block_trans_var, block_trans_sev, False)

    if not np.any(conv):
        warnings.warn("Fixed point not found for all samples")

    if conv_output is True:
        return len_block, conv
    else:
        return len_block

#  LocalWords:  Flyvbjerg