import pandas as pd
from matplotlib.ticker import MaxNLocator

label = [r"$T_{\mathrm{eff}}$", r"$\log{g}$",r"$v_z$", r"$v\sin{i}$", r"$\log{\Omega}$", 
         r"$c^1$", r"$c^2$", r"$c^3$", r"sigAmp", r"logAmp", r"$l$"] 

ws = np.load("../sf/Gl570D/output/marley_grid/run01/temp_emcee_chain.npy")

burned = ws[:, -1000:,:]
xs, ys, zs = burned.shape
fc = burned.reshape(xs*ys, zs)
nx, ny = fc.shape

fig, axes = plt.subplots(11, 1, sharex=True, figsize=(8, 14))
for i in range(0, 11, 1):
    axes[i].plot(burned[:, :, i].T, color="k", alpha=0.2)
    axes[i].yaxis.set_major_locator(MaxNLocator(5))
    axes[i].set_ylabel(label[i])

axes[10].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)

x_vec = np.arange(-1, 1, 0.01)

from numpy.polynomial import Chebyshev as Ch

n_samples, n_dim = fc.shape
n_draws = 900
rand_i = np.random.randint(0, n_samples, size=n_draws)

for i in range(n_draws):

    ch_vec = np.array([0]+list(fc[rand_i[i], 5:7+1]))
    ch_tot = Ch(ch_vec)
    ch_spec = ch_tot(x_vec)

    plt.plot(x_vec, ch_spec, 'r', alpha=0.05)

truth_vals = [810.0, 5.15, 0.0, 30.0] # Saumon et al. 2006;  v_z, and vsini made up from plausible values.

import corner
fig = corner.corner(fc[:, 0:2], labels=label[0:2], show_titles=True, truths=truth_vals[0:2])
fig.savefig('../results/Gl570D_exp1.png', dpi=300)

dat1 = pd.read_csv('../sf/Gl570D/output/marley_grid/run01/spec_config.csv')
dat2 = pd.read_csv('../sf/Gl570D/output/marley_grid/run01/models_draw.csv')

sns.set_style('ticks')

plt.step(dat1.wl, dat1.data, 'k', label='SpeX PRZ')
plt.step(dat1.wl, dat2.model_comp50, 'b', label='draw')
plt.step(dat1.wl, dat1.model_composite, 'r',
         label='\n $T_{\mathrm{eff}}=$810 K, $\log{g}=$5.15')
plt.xlabel('$\lambda \;(\AA)$')
plt.ylabel('$f_\lambda \;(\mathrm{erg/s/cm}^2/\AA)$ ')
plt.title('Gl570D')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('../results/Gl570D_exp1_fit.png', dpi=300, bbox_inches='tight')

plt.step(dat1.wl, dat1.data, 'k', label='SpeX PRZ')
plt.step(dat1.wl, dat2.model_comp50, 'b', label='draw')
plt.step(dat1.wl, dat1.model_composite, 'r',
         label='\n $T_{\mathrm{eff}}=$810 K, $\log{g}=$5.15')
plt.xlabel('$\lambda \;(\AA)$')
plt.ylabel('$f_\lambda \;(\mathrm{erg/s/cm}^2/\AA)$ ')
plt.title('Gl570D')
plt.legend(loc='best')
plt.yscale('linear')

CC = np.load('../sf/Gl570D/output/marley_grid/run01/CC_new.npy')

from scipy.stats import multivariate_normal

#sns.heatmap(CC, xticklabels=False, yticklabels=False)

nz_draw = multivariate_normal(dat2.model_comp50, CC)

plt.figure(figsize=(12, 7))
plt.step(dat1.wl, dat1.data, 'k', label='SpeX PRZ')
plt.plot(dat1.wl, dat2.model_comp50, 'b-', label='draw')

plt.plot(dat1.wl, nz_draw.rvs(), 'g-', label='noise draw')
for i in range(10):
    plt.plot(dat1.wl, nz_draw.rvs(), 'g-', alpha=0.3)

plt.plot(dat1.wl, dat1.model_composite, 'r:',
         label='\n $T_{\mathrm{eff}}=$810 K, $\log{g}=$5.15')
plt.xlabel('$\lambda \;(\AA)$')
plt.ylabel('$f_\lambda \;(\mathrm{erg/s/cm}^2/\AA)$ ')
plt.title('Gl570D with draws from GP')
plt.legend(loc='best')
plt.yscale('linear')

