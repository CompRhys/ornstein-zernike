import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse, median_absolute_error as mae
from scipy.stats import wasserstein_distance as w_dist

def rmse(x_1, x_2):
    return np.sqrt(mse(x_1,x_2))

rdf_aim = np.loadtxt("data/test/rdf_lj_mix_p0.8_n6400_t1.0.dat")
r = rdf_aim[0,:]
rdf_aim = np.mean(rdf_aim[1:,:], axis=0)
r_ibi = np.loadtxt("data/test/down/rdf_ibi_lj-mix_p0.8_b20.0_t1.0.dat")[0,:]
rdf_ibi = np.mean(np.loadtxt("data/test/down/rdf_ibi_lj-mix_p0.8_b20.0_t1.0.dat")[1:,:], axis=0)
rdf_hnc = np.mean(np.loadtxt("data/test/down/rdf_hnc_lj-mix_p0.8_b20.0_t1.0.dat")[1:,:], axis=0)
rdf_nla = np.mean(np.loadtxt("data/test/down/rdf_nla_lj-mix_p0.8_b20.0_t1.0.dat")[1:,:], axis=0)


r_pot, phi_ibi, f_ibi = np.loadtxt("data/test/down/phi_ibi_lj-mix.dat")
_, phi_hnc, f_hnc = np.loadtxt("data/test/down/phi_hnc_lj-mix.dat")
_, phi_nla, f_nla = np.loadtxt("data/test/down/phi_nla_lj-mix.dat")
_, phi_nla_n, f_nla_n = np.loadtxt("data/test/down/phi_nla_n_lj-mix.dat")

print("IBI MSE: {:.4f} MAE: {:.4f} WAS: {:.3f} WASR2: {:.3f}".format(rmse(rdf_aim, rdf_ibi), mae(rdf_aim, rdf_ibi), 
                                                        w_dist(rdf_aim, rdf_ibi), w_dist(r**2*rdf_aim, r**2*rdf_ibi)))
print("HNC MSE: {:.4f} MAE: {:.4f} WAS: {:.3f} WASR2: {:.3f}".format(rmse(rdf_aim, rdf_hnc), mae(rdf_aim, rdf_hnc), 
                                                        w_dist(rdf_aim, rdf_hnc), w_dist(r**2*rdf_aim, r**2*rdf_hnc)))
print("NLA  MSE: {:.4f} MAE: {:.4f} WAS: {:.3f} WASR2: {:.3f}".format(rmse(rdf_aim, rdf_nla), mae(rdf_aim, rdf_nla), 
                                                        w_dist(rdf_aim, rdf_nla), w_dist(r**2*rdf_aim, r**2*rdf_nla)))

matplotlib.rcParams.update({'font.size': 14})

# fig, axes = plt.subplots(1, 3, figsize=(12,3.5))

# axes[0].plot(r_pot, phi_ibi, '-.',label="IBI")
# axes[0].plot(r_pot, phi_hnc, '--',label="HNC")
# axes[0].plot(r_pot, phi_nla, label="NLA")
# axes[0].set_xlim((0,3))
# axes[0].set_ylim((-1.5,3))
# axes[0].set_xlabel('r/$\sigma$')
# axes[0].set_ylabel('$\phi(r)$')
# axes[0].plot(r_pot, phi_nla_n, alpha=0.2, color="tab:green")

# axes[1].plot(r_pot, f_ibi, '-.', label="IBI")
# axes[1].plot(r_pot, f_hnc, '--', label="HNC")
# axes[1].plot(r_pot, f_nla, label="NLA")
# axes[1].set_xlim((0,3))
# axes[1].set_ylim((-12,15))
# axes[1].set_xlabel('r/$\sigma$')
# axes[1].set_ylabel('$f(r)$')
# axes[1].plot(r_pot, f_nla_n, alpha=0.2, color="tab:green")

# axes[2].plot(r_ibi, rdf_ibi, '-.',label="IBI")
# axes[2].plot(r_ibi, rdf_hnc, '--',label="HNC")
# axes[2].plot(r_ibi, rdf_nla, label="NLA")
# axes[2].plot(r, rdf_aim, ':', label="Target")
# axes[2].set_xlim((0,3))
# axes[2].set_xlabel('r/$\sigma$')
# axes[2].set_ylabel('$g(r)$')
# axes[2].legend(markerscale=1, fontsize=12, frameon=False)

# fig.tight_layout()


fig = plt.figure(figsize=(13,5), constrained_layout=False)
gs1 = fig.add_gridspec(nrows=2, ncols=5, left=0.06, right=0.97, bottom=0.12, top=0.95, wspace=0.6, hspace=0.45)
ax0 = fig.add_subplot(gs1[0, 0])
ax1 = fig.add_subplot(gs1[1, 0])
ax2 = fig.add_subplot(gs1[:, 1:3])
ax3 = fig.add_subplot(gs1[:, 3:5])

ax0.plot(r_pot, phi_ibi, '-.', linewidth=2.5, label="IBI")
ax0.plot(r_pot, phi_hnc, '--', linewidth=2.5, label="HNC")
ax0.plot(r_pot, phi_nla, linewidth=2.5, label="NLA")
ax0.set_xlim((0,3))
ax0.set_ylim((-2,3))
ax0.set_xlabel('r/$\sigma$')
ax0.set_ylabel('$\phi(r)$')
ax0.plot(r_pot, phi_nla_n, alpha=0.2, color="tab:green")

ax1.plot(r_pot, f_ibi, '-.', linewidth=2.5, label="IBI")
ax1.plot(r_pot, f_hnc, '--', linewidth=2.5, label="HNC")
ax1.plot(r_pot, f_nla, linewidth=2.5, label="NLA")
ax1.set_xlim((0,3))
ax1.set_ylim((-9,10))
ax1.set_xlabel('r/$\sigma$')
ax1.set_ylabel('$f(r)$')
ax1.plot(r_pot, f_nla_n, alpha=0.2, color="tab:green")

ax0.yaxis.labelpad = ax1.yaxis.labelpad

ax2.plot(r_ibi, rdf_ibi, '-.', linewidth=2.5, label="IBI")
ax2.plot(r_ibi, rdf_hnc, '--', linewidth=2.5, label="HNC")
ax2.plot(r_ibi, rdf_nla, linewidth=2.5, label="NLA")
ax2.plot(r, rdf_aim, ':', linewidth=2.5, label="Target")
ax2.set_xlim((0,3))
ax2.set_xlabel('r/$\sigma$')
ax2.set_ylabel('$g(r)$')
ax2.legend(markerscale=1, fontsize=12, frameon=False)

ax3.plot(r_ibi, rdf_ibi-rdf_aim, '-.', linewidth=2.5, label="IBI")
ax3.plot(r_ibi, rdf_hnc-rdf_aim, '--', linewidth=2.5, label="HNC")
ax3.plot(r_ibi, rdf_nla-rdf_aim, linewidth=2.5, label="NLA")
ax3.plot(r, np.zeros_like(r), ':', linewidth=2.5, label="Target")
ax3.set_xlim((0.8,3))
ax3.set_ylim((-0.4,0.4))
ax3.set_xlabel('r/$\sigma$')
ax3.set_ylabel('$\Delta g(r)$')
ax3.legend(markerscale=1, fontsize=12, frameon=False)

plt.show()