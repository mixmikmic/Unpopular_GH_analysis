get_ipython().magic('matplotlib inline')

import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import snhostspec
from glob import glob

reload(snhostspec)
sim1 = snhostspec.SnanaSimData()
sim1.load_simdata_catalog('wfirst_snhostspec1.cat')
sim1.load_sed_data()

sim1.verbose=2
sim1.simulate_host_spectra(indexlist=[2,4], clobber=True)

sim1.write_catalog('wfirst_snhostspec2.dat')



