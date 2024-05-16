get_ipython().magic('matplotlib inline')
import numpy as np
import healpy as hp
import lsst.sims.maf.db as db
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots

opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'coordinates'

# let's just look at the number of observations in r-band after 2 years with default kwargs
sql = 'filter="r" and night < %i' % (365.25*2)
metric = metrics.CountMetric(col='expMJD')
slicer = slicers.HealpixSlicer()
plotDict = {'colorMax': 75}  # Set the max on the color bar so DD fields don't saturate
plotFuncs = [plots.HealpixSkyMap()] # only plot the sky maps for now
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)

bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)

bg.runAll()
bg.plotAll(closefigs=False)

# Same, only now run at very low resolution
slicer = slicers.HealpixSlicer(nside=8)
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)

slicer = slicers.HealpixSlicer(latCol='ditheredDec', lonCol='ditheredRA')
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)

slicer = slicers.HealpixSlicer(latCol='galb', lonCol='gall')
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)

slicer = slicers.HealpixSlicer(latCol='eclipLat', lonCol='eclipLon')
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)

stacker = stackers.GalacticStacker(raCol='ditheredRA', decCol='ditheredDec')
slicer = slicers.HealpixSlicer(latCol='galb', lonCol='gall')
bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs,
                                    stackerList=[stacker])
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)

slicer = slicers.HealpixSlicer() # back the the default
plotDict = {'colorMax': 75, 'rot':(35, 26, 22.)}

bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)

# Once can also use the healpy display tools:
hp.mollview(bundle.metricValues, max=75)

hp.gnomview(bundle.metricValues, max=75)

hp.cartview(bundle.metricValues, max=75)

hp.orthview(bundle.metricValues, max=75)

slicer = slicers.HealpixSlicer(latCol='zenithDistance', lonCol='azimuth')
plotFuncs=[plots.LambertSkyMap()]
plotDict = {}

bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict=plotDict, plotFuncs=plotFuncs)
bg = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir)
bg.runAll()
bg.plotAll(closefigs=False)

# and this is still a healpix array
hp.mollview(bundle.metricValues, rot=(0,90))



