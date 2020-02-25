import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse, median_absolute_error as mae
from scipy.stats import wasserstein_distance as w_dist

rdf_aim = np.loadtxt("data/test/rdf_lj_mix_p0.8_n6400_t1.0.dat")
r = rdf_aim[0,:]
rdf_aim = np.mean(rdf_aim[1:,:], axis=0)
r_ibi = np.loadtxt("data/test/down/rdf_ibi_lj-mix_p0.8_b20.0_t1.0.dat")[0,:]
rdf_ibi = np.mean(np.loadtxt("data/test/down/rdf_ibi_lj-mix_p0.8_b20.0_t1.0.dat")[1:,:], axis=0)
rdf_hnc = np.mean(np.loadtxt("data/test/down/rdf_hnc_lj-mix_p0.8_b20.0_t1.0.dat")[1:,:], axis=0)
rdf_nla = np.mean(np.loadtxt("data/test/down/rdf_nla_lj-mix_p0.8_b20.0_t1.0.dat")[1:,:], axis=0)


print("IBI MSE: {:.3f} MAE: {:.3f} WAS: {:.3f} WASR2: {:.3f}".format(mse(rdf_aim, rdf_ibi), mse(rdf_aim, rdf_ibi), 
                                                        w_dist(rdf_aim, rdf_ibi), w_dist(r**2*rdf_aim, r**2*rdf_ibi)))
print("HNC MSE: {:.3f} MAE: {:.3f} WAS: {:.3f} WASR2: {:.3f}".format(mse(rdf_aim, rdf_hnc), mae(rdf_aim, rdf_hnc), 
                                                        w_dist(rdf_aim, rdf_hnc), w_dist(r**2*rdf_aim, r**2*rdf_hnc)))
print("NLA  MSE: {:.3f} MAE: {:.3f} WAS: {:.3f} WASR2: {:.3f}".format(mse(rdf_aim, rdf_nla), mae(rdf_aim, rdf_nla), 
                                                        w_dist(rdf_aim, rdf_nla), w_dist(r**2*rdf_aim, r**2*rdf_nla)))

matplotlib.rcParams.update({'font.size': 12})

fig, axes = plt.subplots(1, figsize=(8,6))

axes.plot(r, rdf_aim, label="Target")
axes.plot(r_ibi, rdf_ibi, label="IBI")
axes.plot(r_ibi, rdf_hnc, label="HNC")
axes.plot(r_ibi, rdf_nla,  label="NLA")
axes.set_xlim((0,3))
axes.legend(markerscale=1, fontsize=12, frameon=False)

plt.show()