import numpy as np
import sys
import os 
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# Notation

# r - radius
# rho - density
# h_r - real space total correlation function
# H_k - k-space total correlation function
# c_r - real space direct correlation function
# C_k - k-space direct correlation function

path = os.path.expanduser('~')+'/Liquids'

# test_path = path+'/finite/'
test_path = path+'/finite/output/'
files = sorted_aphanumeric(os.listdir(test_path))
files = files[::-1]

# fig1, axes1 = plt.subplots(2, 2, figsize=(13,13))
fig1, axes1 = plt.subplots(1)
fig2, axes2 = plt.subplots(1)
fig3, axes3 = plt.subplots(1)
fig4, axes4 = plt.subplots(1)
fig5, axes5 = plt.subplots(1)
fig6, axes6 = plt.subplots(1)

for i in range(len(files)):
	output_one = np.loadtxt(test_path+files[i])
	N = re.findall('\d+', files[i])[0]

	r = output_one[:,0]
	phi = output_one[:,1]
	g_r = output_one[:,2]
	sigma_g_r = output_one[:,3]
	c_r = output_one[:,4]
	sigma_c_r = output_one[:,5]
	c_r_fft = output_one[:,7]
	sigma_c_r_fft = output_one[:,8]

	q = output_one[:,6]
	sq = output_one[:,9]
	sq_dir = output_one[:,11]
	sq_fft = output_one[:,12]
	switch = output_one[:,13]
	
	trunc = np.argmax(g_r>0) + 1
	bridge = np.log(g_r[trunc:])+c_r[trunc:]+1.-g_r[trunc:]+phi[trunc:] # nb that ln(0) due to g(r) being zero gives nan which skews the graph

	# axes1.errorbar(r, g_r, sigma_g_r, linewidth=1, errorevery=20, elinewidth=1)
	axes1.plot(r[trunc:], bridge, label= 'N = ' +N)

	if i == 0:
		axes4.plot(r, np.zeros(len(r)), 'b--', linewidth=0.5)

	axes2.plot(r, np.log(r*np.abs(g_r-1.)) + i*4, label= 'N = ' +N)

	axes3.errorbar(r, g_r + i/4., sigma_g_r, linewidth=1, errorevery=20, elinewidth=1, label= 'N = ' +N)

	axes4.errorbar(r, c_r, sigma_c_r, linewidth=1, errorevery=20, elinewidth=1, label= 'N = ' +N)
	
	axes5.errorbar(r, c_r_fft, sigma_c_r_fft, linewidth=1, errorevery=20, elinewidth=1, label= 'N = ' +N)

	# axes6.plot(q, sq, linewidth=1, label= 'S(q)')
	axes6.plot(q[:29], sq_dir[:29], linewidth=2, linestyle='--', label= '$S_{dir}$(q)')
	axes6.plot(q, sq_fft, linewidth=1.5, linestyle='--', label= '$S_{fft}$(q)')
	# axes6.plot(q, switch, linewidth=1.5, color='r', label= 'W(q)')
	
axes1.set_xlabel('$r$')
axes1.set_ylabel('$B(r)$')
axes1.set_xlim([0.75,2])    
axes1.legend(markerscale=3, fontsize = 12)
fig1.tight_layout()

axes2.set_xlabel('$r$')
axes2.set_ylabel('$log(r|g(r)-1|)$')
axes2.legend(markerscale=3, fontsize = 12)
fig2.tight_layout()

# axes3.plot(r, np.ones(len(r)), 'b--', linewidth=0.5)
axes3.set_xlabel('$r$')
axes3.set_ylabel('$g(r)$')
axes3.legend(markerscale=3, fontsize = 12)
fig3.tight_layout()

axes4.set_xlabel('$r$')
axes4.set_ylabel('$c(r)$')
axes4.legend(markerscale=3, fontsize = 12)
fig4.tight_layout()

axes5.set_xlabel('$r$')
axes5.set_ylabel('$c(r)$')
axes5.legend(markerscale=3, fontsize = 12)
fig5.tight_layout()

axes6.set_xlabel('$q$')
axes6.set_ylabel('$S(q)$')
# axes6.set_ylabel('$S(q), W(q)$')
axes6.set_xlim([0,10])
axes6.legend(markerscale=3, fontsize = 12)
fig6.tight_layout()

plt.show()

