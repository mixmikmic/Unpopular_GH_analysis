import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.plots as plots

nside = 32

opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = '2DSlicers'
resultsDb = db.ResultsDb(outDir=outDir)
plotFuncs = [plots.TwoDMap()]

# Plot the total number of visits to each healpixel as a function of time
metric = metrics.AccumulateCountMetric(bins=np.arange(366*10))
slicer = slicers.HealpixSlicer(nside=nside, latCol='ditheredDec', lonCol='ditheredRA')
plotDict = {'colorMax':1000, 'xlabel':'Night (days)'}
sql=''
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)

# Same as above, but now only using OpSim field IDs rather than healpixels
plotFuncs = [plots.TwoDMap()]
slicer = slicers.OpsimFieldSlicer()
plotDict = {'colorMax':1000, 'xlabel':'Night (days)'}
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)

# Make a histogram of the number of visits per field per night in year 1
plotFuncs = [plots.TwoDMap()]
metric = metrics.HistogramMetric(bins=np.arange(367)-0.5)
slicer = slicers.OpsimFieldSlicer()
sql = 'night < 370'
plotDict = {'colorMin':1, 'colorMax':5, 'xlabel':'Night (days)'}
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)

# Now, if we want to see the number of visit pairs, tripples, quads per night, we can just use a different plotter
plotters = [plots.VisitPairsHist()]
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotFuncs=plotters)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)



# Special metrics for computing the co-added depth as a function of time
plotFuncs = [plots.TwoDMap()]
metric = metrics.AccumulateM5Metric(bins=np.arange(365.25*10)-0.5)
slicer = slicers.HealpixSlicer(nside=nside)
sql = 'filter="r"'
plotDict = {'cbarTitle':'mags', 'xlabel':'Night (days)'}
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)

# Look at the minimum seeing as a function of time. 
# One could use this to feed a summary metric to calc when the entire sky has a good template image
# Or, what fraction of the sky has a good template after N years.
plotFuncs = [plots.TwoDMap()]
metric = metrics.AccumulateMetric(col='finSeeing', function=np.minimum,
                                 bins=np.arange(366*10))
slicer = slicers.HealpixSlicer(nside=nside, latCol='ditheredDec', lonCol='ditheredRA')
plotDict = {'xlabel':'Night (days)', 'cbarTitle':'Minimum Seeing (arcsec)', 'colorMax':0.8}
sql='filter="r"'
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)

# Note that these are arrays of healpix maps, so it's easy to pull out a few and plot them all-sky
# I think this means it should be pretty easy to make a matplotlib animation without having to dump each plot to disk: https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
import healpy as hp
for i,night in zip(np.arange(4)+1,[200,400, 800,1200]):
    hp.mollview(bundle.metricValues[:,night], title='night = %i'%night, unit='Seeing (arcsec)', sub=(2,2,i))

# Can use for just a few user-defined points
plotFuncs = [plots.TwoDMap()]
metric = metrics.AccumulateMetric(col='finSeeing', function=np.minimum,
                                 bins=np.arange(366*10))
ra = np.zeros(10.)+np.radians(10.)
dec = np.radians(np.arange(0,10)/9.*(-30))
slicer = slicers.UserPointsSlicer(ra,dec, latCol='ditheredDec', lonCol='ditheredRA')
plotDict = {'xlabel':'Night (days)', 'cbarTitle':'Minimum Seeing (arcsec)', 'colorMax':0.8}
sql='filter="r"'
bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict, plotFuncs=plotFuncs)
group = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir,
                                        resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)









