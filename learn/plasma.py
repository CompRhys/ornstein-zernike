from __future__ import print_function
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from sklearn.metrics import r2_score
from sknn.mlp import Regressor, Layer, Native
from lasagne import layers as lasagne, nonlinearities as nl
import GPy
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from routines.transforms import hr_to_cr, hr_to_sq, sq_to_cr
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

path = os.path.expanduser('~')+'/Liquids/src/learn'

mlp_filename = path+'/ml/mlp.pkl'
mlp_pickle = open(mlp_filename, 'rb')
mlp         = pickle.load(mlp_pickle)
mlp_pickle.close()

mlp_nl_filename   = path+'/ml/nl-mlp.pkl'
mlp_nl_pickle     = open(mlp_nl_filename, 'rb')
mlp_nl            = pickle.load(mlp_nl_pickle)
mlp_nl_pickle.close()

# sgp_filename   = path+'/ml/sgp.pkl'
# sgp_pickle = open(sgp_filename, 'rb')
# sgp         = pickle.load(sgp_pickle)
# sgp_pickle.close()

# sgp_nl_filename   = path+'/ml/nl-sgp.pkl'
# sgp_nl_pickle     = open(sgp_nl_filename, 'rb')
# sgp_nl            = pickle.load(sgp_nl_pickle)
# sgp_nl_pickle.close()

r_min = 0
r_max = 10000.0
samples = np.power(2, 18)
density = 0.75 / np.pi
r = np.linspace(r_min, r_max, samples)
r = r + r[1]
gr = np.zeros_like(r)

Gamma = 24

zeta    = 1.634 + 0.007934*np.sqrt(Gamma) + np.power((1.608/Gamma),2)
sigma   = 1. + 0.01093 * np.power(np.log(Gamma),3)
mu      = 0.246 + 3.145 * np.power(Gamma, 0.75)
nu      = 2.084 + 1.706/np.log(Gamma)
alpha   = 6.908 + np.power((0.860/Gamma), 1./3.)
beta    = 0.231 - 1.785 * np.exp(-Gamma/60.2)
gamma   = 0.140 + 0.215 * np.exp(-Gamma/14.6)
delta   = 3.733 + 2.774 * np.power(Gamma, 1./3.)
epsilon = 0.993 + np.power((33.0/Gamma), 2./3.)

x       = r/zeta - 1.
trunc   = np.argmax(x>0)
Lamda   = (delta - epsilon) * np.exp(-np.sqrt(x[trunc:]/gamma)) + epsilon

gr[:trunc]  = sigma * np.exp(-mu * np.power(-x[:trunc], nu))
gr[trunc:]  = 1. + (sigma - 1.) * np.cos(alpha * x[trunc:] + beta * np.sqrt(x[trunc:])) / np.cosh(x[trunc:] * Lamda)

h           = gr-1
s_int, q    = hr_to_sq(samples, density, h, r, axis=0)

s_low       = np.square(q)/(np.square(q)+3*Gamma)
switch      = 0.5 * (1 + np.tanh((q-1.25)/0.25))

s           = s_int * switch + s_low * (1-switch)

c, rs       = sq_to_cr(samples,density, s, q, axis=0)

phi         = Gamma/r
bridge      = np.log(gr)+c-h+phi

phi_hnc     = (h+1.)-np.log(h+1.)-c-1.

dh          = np.gradient(h,r[1]-r[0])
dc          = np.gradient(c,r[1]-r[0])
py          = np.log(1. + h - c) + c - h

test_set    = np.transpose(np.array((h,c)))
nl_test_set = np.transpose(np.array((h,c,dh,dc)))

# MLP

bridge_mlp  = np.ravel(mlp.predict(test_set))
phi_mlp     = bridge_mlp + h - c - np.log(gr)

bridge_mlp_nl = np.ravel(mlp_nl.predict(nl_test_set))
phi_mlp_nl  = bridge_mlp_nl + h - c - np.log(gr)

# SGP

# b_sgp, var_sgp  = sgp.predict(test_set)
# b_sgp = np.ravel(b_sgp) 
# var_sgp = np.ravel(var_sgp) 
# phi_sgp         = b_sgp + h - c - np.log(gr)

# b_sgp_nl, var_sgp_nl  = sgp_nl.predict(nl_test_set)
# b_sgp_nl = np.ravel(b_sgp_nl) 
# var_sgp_nl = np.ravel(var_sgp_nl) 
# phi_sgp_nl  = b_sgp_nl + h - c - np.log(gr)


fig1, axes1 = plt.subplots(1)
axes1.plot(r, phi, label= 'Simulation') 
axes1.plot(r, phi_mlp,  label= 'Local')
axes1.plot(r, phi_mlp_nl,  label= 'Non-Local')
axes1.plot(r, phi_hnc,  label= 'HNC')
axes1.set_xlim([0,7])  
axes1.set_ylim([0,30])  
axes1.set_xlabel('$r$')
axes1.set_ylabel('$\phi(r)$')
fig1.tight_layout()
axes1.legend(markerscale=3, fontsize = 12)

fig2, axes2= plt.subplots(1)
axes2.plot(r, bridge, label= 'Simulation') 
axes2.plot(r, np.zeros_like(r), label= 'HNC') 
# axes2.plot(r, py,  label= 'PY') 
axes2.plot(r, bridge_mlp,  label= 'Local MLP')
axes2.plot(r, bridge_mlp_nl,  label= 'Non-Local MLP')
# axes2.plot(r, h)
# axes2.plot(r, b_sgp,  label= 'Local (SGP)')
# axes2.fill_between(r, b_sgp-var_sgp, b_sgp+var_sgp, alpha=0.3) 
# axes2.plot(r, b_sgp_nl,  label= 'Non-Local (SGP)')
# axes2.fill_between(r, b_sgp_nl-var_sgp_nl, b_sgp_nl+var_sgp_nl, alpha=0.3)
axes2.set_xlim([0,7]) 
axes2.set_ylim([-3,0.5])  
axes2.set_xlabel('$r$')
axes2.set_ylabel('$B(r)$')
axes2.legend(markerscale=3, fontsize = 12, loc=4)
fig2.tight_layout()

low  = np.argmin(r<1)
high = np.argmin(r<3)

mse_mlp = mae(bridge[low:high], bridge_mlp[low:high])
mse_nl  = mae(bridge[low:high], bridge_mlp_nl[low:high])
mse_hnc = mae(bridge[low:high], np.zeros_like(bridge[low:high]))

# mse_mlp = mse(bridge[low:high], bridge_mlp[low:high])
# mse_nl = mse(bridge[low:high], bridge_mlp_nl[low:high])
# mse_hnc = mse(bridge[low:high], np.zeros_like(bridge[low:high]))

print(mse_hnc, mse_mlp, mse_nl)

# r2_nl  = r2_score(bridge[low:high], bridge_mlp_nl[low:high])
# r2_hnc = r2_score(bridge[low:high], np.zeros_like(bridge[low:high]))
r2_nl  = r2_score(bridge[low:], bridge_mlp_nl[low:])
r2_hnc = r2_score(bridge[low:], np.zeros_like(bridge[low:]))



print(r2_hnc, r2_nl)

# plt.plot(r,gr)


plt.show()

