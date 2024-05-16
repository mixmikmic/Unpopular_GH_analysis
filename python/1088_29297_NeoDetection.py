import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.plots import NeoDistancePlotter
import lsst.sims.maf.plots as plotters

# Set up the database connection
dbDir = '../../tutorials/'
opsdb = db.OpsimDatabase(os.path.join(dbDir,'enigma_1189_sqlite.db'))
outDir = 'NeoDistance_enigma'
resultsDb = db.ResultsDb(outDir=outDir)

slicer = slicers.UniSlicer()
metric = metrics.PassMetric(metricName='NEODistances')
stacker = stackers.NEODistStacker()
stacker2 = stackers.EclipticStacker()
filters = ['u','g','r','i','z','y']

for filterName in filters:
    bundle = metricBundles.MetricBundle(metric, slicer,
                                        'night < 365 and filter="%s"'%filterName,
                                        stackerList=[stacker,stacker2],
                                        plotDict={'title':'%s-band'%filterName},
                                        plotFuncs=[NeoDistancePlotter()])
    bgroup = metricBundles.MetricBundleGroup({filterName:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

metric = metrics.CountMetric('expMJD')
slicer = slicers.HealpixSlicer(nside=64, latCol='eclipLat',lonCol='eclipLon')
stacker = stackers.EclipticStacker(subtractSunLon=True)
sql = ''
plotDict = {'rot':(180,0,0)}
bundle = metricBundles.MetricBundle(metric, slicer,sql,stackerList=[stacker], plotDict=plotDict)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

# Loop through each filter and see the median and max neo dist
metric = metrics.MedianMetric('MaxGeoDist')
slicer = slicers.HealpixSlicer(nside=64, latCol='eclipLat',lonCol='eclipLon')
stacker = stackers.EclipticStacker(subtractSunLon=True)
stacker2 = stackers.NEODistStacker()
plotDict = {'rot':(180,0,0)}
for i,filterName in enumerate(filters):
    sql = 'filter ="%s"'%filterName
    bundle = metricBundles.MetricBundle(metric, slicer,sql,stackerList=[stacker,stacker2], plotDict=plotDict)
    bgroup = metricBundles.MetricBundleGroup({filterName:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

# All observations
metric = metrics.MedianMetric('MaxGeoDist')
metric2 = metrics.MaxMetric('MaxGeoDist')
slicer = slicers.HealpixSlicer(nside=64, latCol='eclipLat',lonCol='eclipLon')
stacker = stackers.EclipticStacker(subtractSunLon=True)
stacker2 = stackers.NEODistStacker()
plotDict = {'rot':(180,0,0)}
bDict = {}
sql = ''
bundle = metricBundles.MetricBundle(metric, slicer,sql,stackerList=[stacker,stacker2], plotDict=plotDict)
bDict[0]=bundle
bundle2 = metricBundles.MetricBundle(metric2, slicer,sql,stackerList=[stacker,stacker2], plotDict=plotDict)
bDict[1] = bundle2
bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

metric = metrics.MaxMetric('solarElong')
metric2 = metrics.MinMetric('solarElong')
slicer = slicers.HealpixSlicer(nside=64)
sql = ''
bundle = metricBundles.MetricBundle(metric, slicer,sql)
bDict = {0:bundle}
bundle = metricBundles.MetricBundle(metric2, slicer,sql)
bDict[2] = bundle
metric = metrics.MedianMetric('solarElong')
bundle = metricBundles.MetricBundle(metric, slicer,sql)
bDict[1]= bundle
bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)



