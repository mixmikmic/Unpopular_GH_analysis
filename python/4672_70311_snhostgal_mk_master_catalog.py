get_ipython().magic('matplotlib inline')

import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import snhostspec
from glob import glob
start = time.time()

reload(snhostspec)
sim = snhostspec.SnanaSimData()
sim.add_all_snana_simdata()

sim.load_matchdata()
sim.pick_random_matches()

sim.load_sed_data()
sim.verbose=2
sim.simulate_host_spectra(clobber=True)

sim.write_catalog("wfirst_snhostspec1.cat")

end = time.time()
print("Finished in {:.1f} sec".format(end-start))

