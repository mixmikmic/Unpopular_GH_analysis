import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import healpy as hp

import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots

stackers.BaseStacker.list(doc=True)

metric = metrics.MeanMetric(col='HA')
slicer = slicers.OneDSlicer(sliceColName='night', binsize=1)
sqlconstraint = 'night<100'
runName = 'enigma_1189'
mB = metricBundles.MetricBundle(metric, slicer, sqlconstraint, runName=runName)

opsdb = db.OpsimDatabase(runName+'_sqlite.db')
outDir = 'stackers_test'
resultsDb = db.ResultsDb(outDir=outDir)

bgroup = metricBundles.MetricBundleGroup({'ha':mB}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
print bgroup.simData.dtype.names

bgroup.plotAll(closefigs=False)

slicer = slicers.HealpixSlicer(nside=64, lonCol='randomDitherFieldNightRa', latCol='randomDitherFieldNightDec')
sqlconstraint = 'filter="r" and night<400'

metric = metrics.CountMetric(col='night')

maxDither = 0.1
stackerList = [stackers.RandomDitherFieldNightStacker(maxDither=maxDither)]

plotDict={'colorMax':50, 'xlabel':'Number of Visits', 'label':'max dither = %.2f' % maxDither}
bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, runName=runName, 
                                    stackerList=stackerList, plotDict=plotDict)
bgroup = metricBundles.MetricBundleGroup({'dither':bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)

bgroup.runAll()

bgroup.plotAll(closefigs=False)

# Update the stacker to use a larger max dither and re-run the bundle
maxDither = 1.75
plotDict2 = {'colorMax':50, 'xlabel':'Number of Visits', 'label':'max dither = %.2f' % maxDither}
stackerList = [stackers.RandomDitherFieldNightStacker(maxDither=maxDither)]
bundle2 = metricBundles.MetricBundle(metric, slicer, sqlconstraint, stackerList=stackerList, plotDict=plotDict2)
bgroup2 = metricBundles.MetricBundleGroup({'dither_large':bundle2}, opsdb, outDir=outDir, resultsDb=resultsDb)

bgroup2.runAll()

bgroup2.plotAll(closefigs=False)

ph = plots.PlotHandler()
ph.setMetricBundles([bundle, bundle2])
ph.setPlotDicts([{'label':'max dither 0.1', 'color':'b'}, {'label':'max dither 1.75', 'color':'r'}])
ph.plot(plots.HealpixPowerSpectrum())



