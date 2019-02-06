import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score

from core.transforms import hr_to_cr, hr_to_sq, sq_to_cr
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

path = os.path.expanduser('~') + '/closure/'

# r_min = 0
# r_max = 10000.0
# samples = np.power(2, 18)
# density = 0.75 / np.pi
# r = np.linspace(r_min, r_max, samples)
# r = r + r[1]
# gr = np.zeros_like(r)

# Gamma = 24

# zeta = 1.634 + 0.007934 * np.sqrt(Gamma) + np.power((1.608 / Gamma), 2)
# sigma = 1. + 0.01093 * np.power(np.log(Gamma), 3)
# mu = 0.246 + 3.145 * np.power(Gamma, 0.75)
# nu = 2.084 + 1.706 / np.log(Gamma)
# alpha = 6.908 + np.power((0.860 / Gamma), 1. / 3.)
# beta = 0.231 - 1.785 * np.exp(-Gamma / 60.2)
# gamma = 0.140 + 0.215 * np.exp(-Gamma / 14.6)
# delta = 3.733 + 2.774 * np.power(Gamma, 1. / 3.)
# epsilon = 0.993 + np.power((33.0 / Gamma), 2. / 3.)

# x = r / zeta - 1.
# trunc = np.argmax(x > 0)
# Lamda = (delta - epsilon) * np.exp(-np.sqrt(x[trunc:] / gamma)) + epsilon

# gr[:trunc] = sigma * np.exp(-mu * np.power(-x[:trunc], nu))
# gr[trunc:] = 1. + (sigma - 1.) * np.cos(alpha * x[trunc:] +
# beta * np.sqrt(x[trunc:])) / np.cosh(x[trunc:] * Lamda)

# h = gr - 1
# s_int, q = hr_to_sq(samples, density, h, r, axis=0)

# s_low = np.square(q) / (np.square(q) + 3 * Gamma)
# switch = 0.5 * (1 + np.tanh((q - 1.25) / 0.25))

# s = s_int * switch + s_low * (1 - switch)

# c, rs = sq_to_cr(samples, density, s, q, axis=0)

# bridge = np.log(gr) + c - h + phi

r = np.array((1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8))
g = np.array((0.031197, 0.50376, 1.83029, 1.92533, 1.245126,
              0.81728, 0.67208, 0.7227, 0.88436))
h = g - 1
c = np.array((-85.9399, -75.0357, -64.3664, -56.3433, -
              50.56, -45.84, -41.84, -38.87, -35.67))
bridge = np.array((-6.2421, -3.8702, -2.0794, -1.0561, -
                   0.5746, -0.3049, -0.2335, -0.0882, 0.0420))

# # MLP

X = np.array((h, c)).T

mlp = load_model(path + 'learn/local.h5')
bridge_mlp = mlp.predict(X)
mlp_r2 = r2_score(bridge, bridge_mlp)
hnc_r2 = r2_score(bridge, np.zeros_like(bridge))

print(mlp_r2, hnc_r2)

# fig1, axes1 = plt.subplots(1)
# axes1.plot(r, phi, label= 'Simulation')
# axes1.plot(r, phi_mlp,  label= 'Local')
# axes1.plot(r, phi_hnc,  label= 'HNC')
# axes1.set_xlim([0,7])
# axes1.set_ylim([0,30])
# axes1.set_xlabel('$r$')
# axes1.set_ylabel('$\phi(r)$')
# fig1.tight_layout()
# axes1.legend(markerscale=3, fontsize = 12)

fig2, axes2 = plt.subplots(1, 3)
axes2[0].plot(r, h, marker='x')
axes2[0].set_xlim([0,7])

axes2[1].plot(r, c, marker='x')
axes2[1].set_xlim([0,7])

axes2[2].plot(r, bridge, label='Simulation')
axes2[2].plot(r, np.zeros_like(r), label='HNC')
axes2[2].plot(r, bridge_mlp,  label='Local MLP')
axes2[2].set_xlim([0,7])
axes2[2].set_ylim([-50,70])
axes2[2].legend()

fig2.tight_layout()


plt.show()
