get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

import os, sys
import numpy as np
from numpy import ma
import xray

dpath = os.path.join(os.environ['HOME'], 'data/NCEP1')

dset_hgt = xray.open_dataset(os.path.join(dpath, 'hgt/hgt.mon.mean.nc'))

dset_hgt

lat = dset_hgt['lat'].data
lon = dset_hgt['lon'].data

dset_hgt = dset_hgt.sel(time=slice('1948','2014'))

dates = dset_hgt['time'].data

hgt_700 = dset_hgt.sel(level=700)

hgt_700

hgt_700 = hgt_700.sel(lat=slice(-20,-90.))

lat = hgt_700['lat'].data
lon = hgt_700['lon'].data

# hgt_700.close()

hgt_700

def demean(x): 
    return x - x.sel(time=slice('1981-1-1','2010-12-1')).mean('time')

hgt_700_anoms = hgt_700.groupby('time.month').apply(demean)



from eofs.standard import Eof

coslat = np.cos(np.deg2rad(lat))
wgts = np.sqrt(coslat)[..., np.newaxis]

X = hgt_700_anoms['hgt'].data

X = ma.masked_array(X)

solver = Eof(X, weights=wgts)

eof1 = solver.eofsAsCorrelation(neofs=1)

pc1 = solver.pcs(npcs=1, pcscaling=1)

plt.plot(pc1)

eof1.shape

plt.imshow(eof1.squeeze())

from matplotlib.mlab import detrend_linear

dpc1 = detrend_linear(pc1.squeeze())

plt.plot(dpc1)

time = hgt_700_anoms.time.to_index()

import pandas as pd

SAM = pd.DataFrame(dpc1, index=time, columns=['SAM'])

SAM.to_csv('../data/SAM_index_1948_2014_1981_2010_Clim.csv')

