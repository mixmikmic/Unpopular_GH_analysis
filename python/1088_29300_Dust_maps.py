import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import numpy as np

# Set up the database connection
opsdb = db.OpsimDatabase('minion_1016_sqlite.db')
outDir = 'dustMaps'
resultsDb = db.ResultsDb(outDir=outDir)

nside=128
bundleList = []
slicer1 = slicers.HealpixSlicer(nside=nside)
slicer2 = slicers.HealpixSlicer(nside=nside, useCache=False)
metric1 = metrics.Coaddm5Metric()
metric2 = metrics.ExgalM5()
filters = ['u', 'g', 'r', 'i', 'z', 'y']
mins = {'u': 23.7, 'g':25.2, 'r':22.5, 'i': 24.6, 'z': 23.7,'y':22.8 }
maxes = {'u': 27.6, 'g':28.5, 'r': 28.5, 'i': 27.9, 'z': 27.6,'y':26.1 }

for filtername in filters:
    sql = 'filter="%s"' % filtername
    plotDict = {'colorMin': mins[filtername], 'colorMax': maxes[filtername]}
    bundleList.append(metricBundles.MetricBundle(metric1, slicer1, sql))
    bundleList.append(metricBundles.MetricBundle(metric2, slicer2, sql, plotDict=plotDict))

bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

bgroup.plotAll(closefigs=False)



