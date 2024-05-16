import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns

from astropy.io import fits
import h5py

with open('../data/raw/2M_J0050.pic', 'rb') as f:
    wlgrid, Flux, Flux_err = np.load(f, allow_pickle=True, encoding='bytes')

plt.step(wlgrid, Flux, 'k', where='mid')
plt.fill_between(wlgrid, Flux-Flux_err, Flux+Flux_err, color='r', step='mid')
plt.xlabel('$λ\;(μ\mathrm{m})$')
plt.ylabel('$f_λ \;(\mathrm{W/m}^2/\mathrm{m})$')

with open('../data/raw/2M_J0415.pic', 'rb') as f:
    wlgrid, Flux, Flux_err = np.load(f, allow_pickle=True, encoding='bytes')

plt.step(wlgrid, Flux, 'k', where='mid')
plt.fill_between(wlgrid, Flux-Flux_err, Flux+Flux_err, color='r', step='mid')
plt.xlabel('$λ\;(μ\mathrm{m})$')
plt.ylabel('$f_λ \;(\mathrm{W/m}^2/\mathrm{m})$')

with open('../data/raw/HD3651B.pic', 'rb') as f:
    wlgrid, Flux, Flux_err = np.load(f, allow_pickle=True, encoding='bytes')

plt.step(wlgrid, Flux, 'k', where='mid')
plt.fill_between(wlgrid, Flux-Flux_err, Flux+Flux_err, color='r', step='mid')
plt.xlabel('$λ\;(μ\mathrm{m})$')
plt.ylabel('$f_λ \;(\mathrm{W/m}^2/\mathrm{m})$')

len(wlgrid)

