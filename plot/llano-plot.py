import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

# Notation

# r - radius
# rho - density
# h_r - real space total correlation function
# H_k - k-space total correlation function
# c_r - real space direct correlation function
# C_k - k-space direct correlation function

# read data 
path = os.path.expanduser('~')+'/Liquids'
test_number = str(22001)
output_fp = np.loadtxt(path+'/llano/output/output_'+test_number+'.dat')

r = output_fp[:,0]
phi = output_fp[:,1]
g_r = output_fp[:,2]
err_g_r = output_fp[:,3]
c_r = output_fp[:,4]
err_c_r = output_fp[:,5]
cfft = output_fp[:,7]
err_cfft = output_fp[:,8]


rllano = np.arange(0.05, 2.55, 0.05)

llano = np.array((-3.7631,-3.5844,-3.3491,-3.1477,-2.9082,-2.6639,-2.4605,-2.2480,-2.0415,
-1.8444,-1.6425,-1.4807,-1.2964,-1.1140,-0.9544,-0.8064,-0.6634,-0.5510,-0.4425,-0.3044,
-0.1995,-0.1312,-0.0855,-0.0460,-0.0241,-0.0075,0.0000,0.0023,
-0.0030,-0.0062,-0.0152,-0.0183,-0.0216,-0.0214,-0.0178,-0.0095,-0.0060,0.0007,0.0027,0.00256,
-0.0030,-0.0032,-0.0038,-0.0057,-0.0060,-0.0039,-0.0041,-0.0039,-0.0046,-0.0100))

# llano = np.array((-0.8924,-0.8692,-0.8132,-0.7585,-0.6901,-0.6251,-0.5703,-0.5032,-0.4370,-0.4051,
# -0.3345,-0.3114,-0.2711,-0.2134,-0.2247,-0.1658,-0.1502,-0.1172,-0.0731,-0.0734,
# -0.0631,-0.0527,-0.0164,-0.0204,-0.0195,-0.0016,-0.0156,-0.0154,-0.0159,-0.0288,
# -0.0418,-0.0155,-0.0169,-0.0209,-0.0192,-0.0177,-0.0293,-0.0296,-0.0027,0.0014,
# -0.0012,-0.0073,0.0003,-0.0019,0.0080,0.0085,0.0027,0.0123,0.0057,0.0079))

llano = np.array((-6.8757,-6.5826,-6.1928,-5.7682,-5.3394,-4.9202,-4.5109,-4.1127,-3.7278,-3.3629,
-2.9926,-2.6308,-2.292,-1.9863,-1.6845,-1.4042,-1.1465,-0.8964,-0.692,
-0.5205,-0.3708,-0.25,-0.1581,-0.0912,-0.0477,-0.0242,-0.0173,-0.0122,-0.0154,
-0.0239,-0.0362,-0.0405,-0.0373,-0.0395,-0.0251,-0.0193,-0.0085,-0.0011,-0.0043,
-0.0091,0.0196,0.0192,0.0217,0.0211,0.0164,0.0148,0.0144,0.0123,
-0.0176,-0.0223))

try:
    trunc_zero = np.max(np.where(g_r==0.0))+1
except ValueError:
    trunc_zero = 0
if int(test_number[0]) == 4:
    trunc_zero = np.min(np.where(g_r>1.0))+1
trunc_small = 5
trunc = max(trunc_small, trunc_zero)

bridge = np.log(g_r[trunc:])+c_r[trunc:]+1.-g_r[trunc:]+phi[trunc:]
err_bridge = np.sqrt(np.square((g_r[trunc:]-1.)*err_g_r[trunc:]/g_r[trunc:])+np.square(err_c_r[trunc:]))

bridge_fft = np.log(g_r[trunc:])+cfft[trunc:]+1.-g_r[trunc:]+phi[trunc:]
err_bridge_fft = np.sqrt(np.square((g_r[trunc:]-1.)*err_g_r[trunc:]/g_r[trunc:])+np.square(err_cfft[trunc:]))


fig, axes = plt.subplots()
axes.errorbar(r[trunc:], c_r[trunc:], err_c_r[trunc:], linewidth=1, errorevery=5, ecolor='r', elinewidth=1)
axes.errorbar(r[trunc:], cfft[trunc:], err_cfft[trunc:], linewidth=1, errorevery=5, ecolor='r', elinewidth=1)
axes.plot(r, np.zeros(len(r)), 'b--', linewidth=0.5)
axes.set_xlabel('$r$')
axes.set_ylabel('$c(r)$')

fig, axes = plt.subplots()
axes.errorbar(r[trunc:], bridge, err_bridge, color='orange', linewidth=1,  ecolor='r', elinewidth=1, Label='Present Work')
# axes.errorbar(r[trunc:], bridge_fft, err_bridge_fft, color='orange', linewidth=1,  ecolor='r', elinewidth=1)
axes.plot(rllano, llano, linestyle="None", marker='x', label='Llano-Restrepo')
axes.plot(r, np.zeros(len(r)), 'b--', linewidth=0.5)
axes.set_xlabel('$r$')
axes.set_ylabel('$B(r)$')
# axes.set_ylim([-0.4,0.05]) 
axes.set_xlim([0.75,2])
axes.legend(markerscale=1, fontsize = 12)
# plt.savefig(path+'/figures/gr'+test_number+'.png')

plt.show()
