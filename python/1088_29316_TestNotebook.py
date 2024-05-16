print "Hi"

import os
yourname = os.getenv('USER')

print "Hi", yourname

currentdir = os.getenv('PWD')
print "I think I'm in directory ", currentdir

filenames = get_ipython().getoutput('ls *ipynb')
print "Notebooks in this directory include: "
print filenames

import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plot
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metricBundles as metricBundles



