from __future__ import print_function
import sys
import os 
import numpy as np
from routines import flyvbjerg, autocorr
import matplotlib.pyplot as plt
import timeit

# test error methods for autoregressive model

n_samples = [64,128,256,512,1024,2048,4096,8192]
n_tests = 500

errors = np.zeros_like(n_samples, dtype=float)
errors_flyv = np.zeros_like(n_samples, dtype=float)
errors_tau = np.zeros_like(n_samples, dtype=float)
times = np.zeros_like(n_samples, dtype=float)

for i in range(len(n_samples)):
	a = 0.9
	x = w = np.random.normal(size=(n_samples[i],n_tests))
	for t in range(n_samples[i]):
	    x[t] = (a*x[t-1] + w[t])/(1+a)

	data = x + 5.

	# Normal Error
	err = np.std(data, axis=0, ddof=1)

	# Flyvberg Method 1
	start = timeit.default_timer()
	err_flyv = flyvbjerg.fp_stderr(data)
	stop = timeit.default_timer()
	time_flyv = stop-start

	# Correlation Time Method
	start = timeit.default_timer()
	err_tau = autocorr.ac_stderr(data)
	stop = timeit.default_timer()
	time_tau = stop-start

	errors[i] = np.sum(err)
	errors_flyv[i] = np.sum(err_flyv)*np.sqrt(n_samples[i])
	errors_tau[i] = np.sum(err_tau)*np.sqrt(n_samples[i])
	times[i] = time_tau/time_flyv

plt.figure(1)
plt.title('flybjerg vs tau methods error in a point')
plt.plot(n_samples, errors_tau, 'r', label='Autocorrelation Time')
plt.plot(n_samples, errors_flyv, label='Block Averaging')
plt.plot(n_samples, errors, 'g', label='Standard deviation')

# plt.figure(2)
# plt.title('flybjerg vs tau methods ratio of errors')
# plt.plot(n_samples, errors_flyv/errors_tau)
# plt.figure(3)
# plt.title('flybjerg vs tau methods ratio of times')
# plt.plot(n_samples, times)
plt.legend()
plt.tight_layout()
plt.show()
