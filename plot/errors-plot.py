import numpy as np
import sys
import matplotlib.pyplot as plt

# Notation

# r - radius
# rho - density
# h_r - real space total correlation function
# H_k - k-space total correlation function
# c_r - real space direct correlation function
# C_k - k-space direct correlation function

# read data 
path = os.path.expanduser('~')+'/Liquids'
test_number = str(input("Input file number: "))
output_fp = np.loadtxt(path+'/data/output/errors_'+test_number+'.dat')

r = output_fp[:,0]
sigma = output_fp[:,1]
sigma_fp = output_fp[:,3]
sigma_ac = output_fp[:,4]
# fsigma = output_fp[:,2]
# fsigma_fp = output_fp[:,5]
# fsigma_ac = output_fp[:,6]

print(np.sum(sigma_ac)/np.sum(sigma_fp))
# # Plot

plt.figure(1)
plt.plot(r, sigma, label='Standard deviation')
plt.plot(r, sigma_ac, label='Autocorrelation Time')
plt.plot(r, sigma_fp, label='Block Averaging')
plt.savefig(path+'/figures/err'+test_number+'.png')

# plt.figure(2)
# plt.plot(r, fsigma, color='g')
# plt.plot(r, fsigma_ac, color='r')
# plt.plot(r, fsigma_fp)
# plt.savefig(path+'/figures/ferr'+test_number+'.png')

plt.show()
