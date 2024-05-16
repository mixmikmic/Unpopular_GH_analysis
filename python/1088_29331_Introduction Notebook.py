# Check the version of MAF - the output should be version 1.1.1 or higher.
import lsst.sims.maf
lsst.sims.maf.__version__

# import matplotlib to show plots inline.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# import our python modules
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles

# metric = the "maximum" of the "airmass" for each group of visits in the slicer
metric1 = metrics.MaxMetric('airmass')

# slicer = a grouping or subdivision of visits for the simulated survey based on their position on the sky 
# (using a Healpix grid)
slicer1 = slicers.HealpixSlicer(nside=64)

# sqlconstraint = the sql query (or 'select') that selects all visits in r band.
sqlconstraint= 'filter = "r"'

# MetricBundle = combination of the metric, slicer, and sqlconstraint
maxairmassSky = metricBundles.MetricBundle(metric1, slicer1, sqlconstraint)

opsdb = db.OpsimDatabase('ops2_1114_sqlite.db')
outDir = 'output_directory'
resultsDb = db.ResultsDb(outDir=outDir)

bundleDict = {'maxairmassSky':maxairmassSky}

group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
group.runAll()

group.plotAll(closefigs=False)

metric2 = metrics.CountMetric('expMJD')
nvisitsSky = metricBundles.MetricBundle(metric2, slicer1, sqlconstraint)

summaryMetrics = [metrics.MinMetric(), metrics.MedianMetric(), metrics.MaxMetric(), metrics.RmsMetric()]
maxairmassSky.setSummaryMetrics(summaryMetrics)
nvisitsSky.setSummaryMetrics(summaryMetrics)

# A slicer that will calculate a metric after grouping the visits into subsets corresponding to each night.
slicer2 = slicers.OneDSlicer(sliceColName='night', binsize=1, binMin=0, binMax=365*10)

# We can combine these slicers and metrics and generate more metricBundles
nvisitsPerNight = metricBundles.MetricBundle(metric1, slicer2, sqlconstraint, summaryMetrics=summaryMetrics)
maxairmassPerNight = metricBundles.MetricBundle(metric2, slicer2, sqlconstraint, summaryMetrics=summaryMetrics)

bundleDict = {'maxairmassSky':maxairmassSky, 'maxairmassPerNight':maxairmassPerNight, 
        'nvisitsSky':nvisitsSky, 'nvisitsPerNight':nvisitsPerNight}

group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)

print "Array with the number of visits per pixel:", nvisitsSky.metricValues

import numpy as np
np.max(nvisitsSky.metricValues)

print "Summary of the max, median, min, and rms of the number of visits per pixel", nvisitsSky.summaryValues

# This is what the skymap should look like:
from IPython.display import Image
Image(filename='images/thumb.ops2_1114_Mean_finSeeing_i_HEAL_SkyMap.png')



