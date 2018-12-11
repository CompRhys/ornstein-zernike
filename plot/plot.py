import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Notation

# r - radius
# rho - density
# h_r - real space total correlation function
# H_k - k-space total correlation function
# c_r - real space direct correlation function
# C_k - k-space direct correlation function

# read data 
path = os.path.expanduser('~')+'/masters/closure'
test_number = str(input("Input file number: "))
output_fp = np.loadtxt(path+'/data/old/output/output_'+test_number+'.dat')

r = output_fp[:,0]
phi = output_fp[:,1]
g_r = output_fp[:,2]
err_g_r = output_fp[:,3]
c_r = output_fp[:,4]
cfft = output_fp[:,7]
err_c_r = output_fp[:,5]
err_cfft = output_fp[:,8]
q = output_fp[:,6]
s_q = output_fp[:,9]
sigma_s_q = output_fp[:,10]
# s_dir = output_fp[:,11]
# s_fft = output_fp[:,12]
# w_q = output_fp[:,13]

try:
    trunc_zero = np.max(np.where(g_r==0.0))+1
except ValueError:
    trunc_zero = 0
if int(test_number[0]) == 4:
    trunc_zero = np.min(np.where(g_r>1.0))+1
trunc_small = 5
trunc = max(trunc_small, trunc_zero)

# # backward estimates of phi

# py_bridge = np.log(g_r-c_r)+c_r-g_r+1

bridge = np.log(g_r[trunc:])+c_r[trunc:]+1.-g_r[trunc:]+phi[trunc:]
err_bridge = np.sqrt(np.square((g_r[trunc:]-1.)*err_g_r[trunc:]/g_r[trunc:])+np.square(err_c_r[trunc:]))

bridge_fft = np.log(g_r[trunc:])+cfft[trunc:]+1.-g_r[trunc:]+phi[trunc:]
err_bridge_fft = np.sqrt(np.square((g_r[trunc:]-1.)*err_g_r[trunc:]/g_r[trunc:])+np.square(err_cfft[trunc:]))

# bridge_min = np.argmin(bridge)
# bridge = bridge[bridge_min:]
# err_bridge = err_bridge[bridge_min:]
# trunc += bridge_min

phi_approx_hnc = (g_r[trunc:]-np.log(g_r[trunc:])-c_r[trunc:]-1.)
err_approx_hnc = np.sqrt(np.square((g_r[trunc:]-1.)*err_g_r[trunc:]/g_r[trunc:])+np.square(err_c_r[trunc:]))

# phi_approx_py = np.log(1.-c_r/g_r)
# err_approx_py = np.sqrt(np.square(1./(c_r-g_r))*(np.square(c_r*sigma_g_r/g_r)+np.square(sigma_c_r)))


# # Plot

fig, axes = plt.subplots(2, figsize=(10,8))
axes[0].errorbar(r, g_r, err_g_r, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
axes[0].plot(r, np.ones(len(r)), 'b--', linewidth=0.5)
axes[0].set_xlabel('r/$\sigma$')
axes[0].set_ylabel('$g(r)$')
# axes[0].set_xlim([0,2]) 
axes[1].errorbar(r, c_r, err_c_r, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
axes[1].plot(r, cfft, 'g', linewidth=1)
axes[1].plot(r, np.zeros(len(r)), 'b--', linewidth=0.5)
axes[1].set_xlabel('r/$\sigma$')
axes[1].set_ylabel('$c(r)$')
# axes[1,0].plot(r, phi, linewidth=2)
# axes[1,0].errorbar(r, phi_approx_py, err_approx_py, 
# 	color='orange', linewidth=2, errorevery=20, ecolor='r', elinewidth=1)
# axes[1,0].set_ylim([-5,5])    
# axes[1,0].set_xlabel('r/$\sigma$')
# axes[1,0].set_ylabel('$\phi(r)$ [py]')
# plt.savefig(path+'/figures/corr'+test_number+'.png')


# fig, axes = plt.subplots(2)
# axes[0].errorbar(r, g_r, err_g_r, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# axes[0].plot(r, np.ones(len(r)), 'b--', linewidth=0.5)
# axes[0].set_xlabel('$r$')
# axes[0].set_ylabel('$g(r)$')
# axes[0].set_xlim([0,5]) 
# axes[1].errorbar(q, s_q, sigma_s_q, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# axes[1].set_xlabel('$q$')
# axes[1].set_ylabel('$S(q)$')
# axes[1].set_xlim([0,10]) 
# # plt.savefig(path+'/figures/corr'+test_number+'.png')


# fig, axes = plt.subplots(2)
# axes[0].errorbar(r[trunc:], bridge, err_bridge, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# axes[0].errorbar(r[trunc:], bridge_fft, err_bridge_fft, linewidth=1, color='g', errorevery=20, ecolor='r', elinewidth=1)
# # axes[0].plot(r, py_bridge, 'b--', linewidth=0.5)
# axes[0].plot(r, np.zeros(len(r)), 'b--', linewidth=0.5)
# axes[0].set_xlabel('r/$\sigma$')
# axes[0].set_ylabel('$B(r)$')
# # axes[0].set_ylim([-1,1]) 
# # axes[0].set_xlim([0,2]) 
# axes[1].plot(r, phi, linewidth=2)
# axes[1].errorbar(r[trunc:], phi_approx_hnc, err_approx_hnc, 
# 	color='orange', linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# axes[1].set_ylim([-15,15])
# # axes[1].set_xlim([0,2])    
# axes[1].set_xlabel('r/$\sigma$')
# axes[1].set_ylabel('$\phi(r)$ [hnc]')
# # plt.savefig(path+'/figures/bridge'+test_number+'.png')


# # fig, axes = plt.subplots(1)
# # # axes.plot(q, s_q, linewidth=1)
# # axes.plot(q, s_dir, linewidth=1)
# # axes.plot(q, s_fft, linewidth=1)
# # axes.plot(q, w_q, linewidth=1)
# # # axes.errorbar(q, s_q, sigma_s_q, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# # # axes.set_xlim([0,50])  
# # axes.set_xlabel('$q$')
# # axes.set_ylabel('$S(q)$')
# # plt.savefig(path+'/figures/struct'+test_number+'.png')

# fig, axes = plt.subplots(1, figsize=(10,10))
# axes.plot(r, np.log(r*np.abs(g_r-1.)))
# axes.set_xlabel('$r$')
# axes.set_ylabel('$log(r*|g(r)-1|)$')
# plt.savefig(path+'/figures/gr'+test_number+'.png')

plt.show()
