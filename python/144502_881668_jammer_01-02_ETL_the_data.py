import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns

from astropy.io import fits
import h5py

import astropy.units as u

from os import listdir

files = listdir('../data/raw/')

for i, file in enumerate(files):
    with open('../data/raw/{}'.format(file), 'rb') as f:
        wlgrid, Flux, Flux_err = np.load(f, allow_pickle=True, encoding='bytes')
        
    out_name = '../data/reduced/{}.hdf5'.format(file[:-4])
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
    print("{:03d}: {:.0f}  -  {:.0f}   {}".format(i, wls_out[0], wls_out[-1], out_name))
    f_new.close()

