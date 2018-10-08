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
path = os.path.expanduser('~')+'/Liquids'
test_file = raw_input("Input file type: ")
test_number = raw_input("Input file number: ")
if test_file == 'solid':
	output_fp = np.loadtxt(path+'/data/discard/solid/output_'+test_number+'.dat')
elif test_file == 'temp':
	output_fp = np.loadtxt(path+'/data/discard/temp/output_'+test_number+'.dat')
elif test_file == 'two':
	output_fp = np.loadtxt(path+'/data/discard/two-phase/output_'+test_number+'.dat')
else:
	output_fp = np.loadtxt(path+'/data/discard/unexplained/output_'+test_number+'.dat')


r = output_fp[:,0]
phi = output_fp[:,1]
g_r = output_fp[:,2]
sigma_g_r = output_fp[:,3]
c_r = output_fp[:,4]
sigma_c_r = output_fp[:,5]
q = output_fp[:,6]
s_q = output_fp[:,9]
sw = output_fp[:,11]
# sigma_s_q = output_fp[:,8]

# # backward estimates of phi

# phi_approx_py = np.log(1.-c_r/g_r)
# err_approx_py = np.sqrt(np.square(1./(c_r-g_r))*(np.square(c_r*sigma_g_r/g_r)+np.square(sigma_c_r)))

phi_approx_hnc = (g_r-np.log(g_r)-c_r-1.)
err_approx_hnc = np.sqrt(np.square((g_r-1.)*sigma_g_r/g_r)+np.square(sigma_c_r))

bridge = np.log(g_r)+c_r+1.-g_r+phi # nb that ln(0) due to g(r) being zero gives nan which skews the graph
err_bridge = np.sqrt(np.square((g_r-1.)*sigma_g_r/g_r)+np.square(sigma_c_r))

# # Plot

# fig, axes = plt.subplots(2, figsize=(10,10))
# axes[0].errorbar(r, g_r, sigma_g_r, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# axes[0].plot(r, np.ones(len(r)), 'b--', linewidth=0.5)
# axes[0].set_xlabel('r/$\sigma$')
# axes[0].set_ylabel('$g(r)$')
# # axes[0].set_xlim([0,2]) 
# axes[1].errorbar(r, c_r, sigma_c_r, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# axes[1].plot(r, np.zeros(len(r)), 'b--', linewidth=0.5)
# axes[1].set_xlabel('r/$\sigma$')
# axes[1].set_ylabel('$c(r)$')
# plt.savefig(path+'/figures/corr_'+test_number+'.png')

fig, axes = plt.subplots()
axes.plot(q, s_q, linewidth=1)
# axes.plot(q, np.ones(len(q))*2.8, color='r', linestyle='--')
axes.set_xlabel('$q$')
axes.set_ylabel('$S(q)$')
axes.set_xlim([0,20])
# axes.set_ylim([,])
plt.savefig(path+'/figures/struct_'+test_number+'.png')

# fig, axes = plt.subplots(2, figsize=(10,10))
# axes[0].errorbar(r, bridge, err_bridge, linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# axes[0].plot(r, np.zeros(len(r)), 'b--', linewidth=0.5)
# axes[0].set_xlabel('r/$\sigma$')
# axes[0].set_ylabel('$B(r)$')
# # axes[0].set_ylim([-1,1]) 
# # axes[0].set_xlim([0,2]) 
# axes[1].plot(r, phi, linewidth=2)
# axes[1].errorbar(r, phi_approx_hnc, err_approx_hnc, 
# 	color='orange', linewidth=1, errorevery=20, ecolor='r', elinewidth=1)
# axes[1].set_ylim([-15,15])
# # axes[1].set_xlim([0,2])    
# axes[1].set_xlabel('r/$\sigma$')
# axes[1].set_ylabel('$\phi(r)$ [hnc]')
# plt.savefig(path+'/figures/bridge_'+test_number+'.png')



# fig, axes = plt.subplots(1, figsize=(10,10))
# axes.plot(r, np.log(r*np.abs(g_r-1.)))
# axes.set_xlabel('$r$')
# axes.set_ylabel('$log(r*|g(r)-1|)$')
# plt.savefig(path+'/figures/gr_'+test_number+'.png')

plt.show()
