import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
from lsst.sims.maf.plots import PlotHandler

# Set the database and query
database = 'pontus_1074.db'
sqlWhere = 'filter = "r" and night < 400'
opsdb = db.OpsimDatabase(database)
outDir = 'slicers_test'
resultsDb = db.ResultsDb(outDir=outDir)

# For the count metric the col kwarg is pretty much irrelevant, so we'll just use expMJD, but any column in the database would work
metric = metrics.CountMetric(col='observationStartMJD', metricName='Count')

slicer = slicers.UniSlicer()

bundles = {}
bundles['uni'] = metricBundles.MetricBundle(metric,slicer,sqlWhere)

slicer = slicers.OneDSlicer(sliceColName='night', binsize=10)
bundles['oneD'] = metricBundles.MetricBundle(metric,slicer,sqlWhere)

slicer = slicers.HealpixSlicer(nside=64)
metric2 = metrics.Coaddm5Metric()
bundles['healpix'] = metricBundles.MetricBundle(metric2,slicer,sqlWhere)

slicer = slicers.OpsimFieldSlicer()
bundles['ops'] = metricBundles.MetricBundle(metric,slicer,sqlWhere)

bgroup = metricBundles.MetricBundleGroup(bundles,opsdb, outDir=outDir, resultsDb=resultsDb)

bgroup.runAll()
bgroup.plotAll(closefigs=False)

print bundles['uni'].metricValues
bundles['uni'].plot()

print bundles['oneD'].metricValues
bundles['oneD'].plot()

print bundles['healpix'].metricValues
bundles['healpix'].setPlotDict({'colorMin':0, 'colorMax':50})
bundles['healpix'].plot()





