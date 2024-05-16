import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns

from astropy.io import fits
import h5py

import astropy.units as u

from os import listdir

file = '../../../other_GitHub/splat/reference/Spectra/10020_11100.fits'

fits_dat = fits.open(file)
fd_0 = fits_dat[0]
fd_0.data.shape

with open('../data/raw/Gl570D.pic', 'rb') as f:
    wlgrid_orig, Flux_orig, Flux_err_orig = np.load(f, allow_pickle=True, encoding='bytes')

plt.step(fd_0.data[0, :], fd_0.data[1, :]*np.max(Flux_orig))
plt.step(fd_0.data[0, :], fd_0.data[2, :]*np.max(Flux_orig))
plt.plot(wlgrid_orig, Flux_orig, '-')

plt.plot(fd_0.data[0, :],  fd_0.data[1, :]/fd_0.data[2, :], 'o')
plt.ylabel('$\sigma ?$')

wlgrid = fd_0.data[0, :]
Flux = fd_0.data[1, :]*np.max(Flux_orig)
Flux_err = np.abs(fd_0.data[1, :]/fd_0.data[2, :])*np.max(Flux_orig)

plt.step(wlgrid, Flux)
plt.plot(wlgrid, Flux_err)

plt.plot(wlgrid, Flux/Flux_err, '.')

bi = ((Flux/Flux_err) > 75)

bi.sum()

Flux_err[bi] = Flux[bi]/75.0

bi2 = np.abs(Flux_err) == np.inf
Flux_err[bi2] = np.abs(Flux[bi2]*3.0)

out_name = '../data/reduced/Gl570D_full.hdf5'
fls_out = (Flux*u.Watt/u.m**2/u.m).to(u.erg/u.s/u.cm**2/u.Angstrom).value
sig_out = (Flux_err*u.Watt/u.m**2/u.m).to(u.erg/u.s/u.cm**2/u.Angstrom).value
#print(out_name, np.min(sig_out), np.sum(sig_out==0), np.percentile(fls_out/sig_out, 80))
bi = sig_out <= 0
sig_out[bi] = np.abs(fls_out[bi])
wls_out = wlgrid*10000.0
msk_out = np.ones(len(wls_out), dtype=int)
f_new = h5py.File(out_name, 'w')
f_new.create_dataset('fls', data=fls_out)
f_new.create_dataset('wls', data=wls_out)
f_new.create_dataset('sigmas', data=sig_out)
f_new.create_dataset('masks', data=msk_out)
print("{:.0f}  -  {:.0f}   {}".format(wls_out[0], wls_out[-1], out_name))
f_new.close()

