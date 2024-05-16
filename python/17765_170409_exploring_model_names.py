get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import os; 
os.chdir('..')
os.chdir('..')
import models
import datasources
import variables
import orca
import pandas as pd
import numpy as np

orca.run([
    "neighborhood_vars",
    "rsh_simulate",
    "rrh_simulate",
    "price_vars",
], iter_vars=[2010])

hh = orca.get_table('households').to_frame()
hh[1:5]



