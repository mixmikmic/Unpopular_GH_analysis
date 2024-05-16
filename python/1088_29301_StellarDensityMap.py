import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.maps as maps

# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'starMap_test'
resultsDb = db.ResultsDb(outDir=outDir)

bundleList = []
sql = 'night < %i' % (365.25*3) # See How well we do after year 3
slicer = slicers.HealpixSlicer(nside=64, useCache=False)
metric = metrics.StarDensityMetric(metricName='rmag<25')
# setup the stellar density map to use. By default, all stars in the CatSim catalog are included
mafMap = maps.StellarDensityMap()
plotDict = {'colorMin':0.001, 'colorMax':.1, 'logScale':True}
bundle = metricBundles.MetricBundle(metric,slicer,sql, mapsList=[mafMap], plotDict=plotDict)
bundleList.append(bundle)

metric = metrics.StarDensityMetric(rmagLimit=27.5,metricName='rmag<28')
bundle = metricBundles.MetricBundle(metric,slicer,sql, mapsList=[mafMap], plotDict=plotDict)
bundleList.append(bundle)
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)

bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

bundleList = []
sql = 'night < %i' % (365.25*3) # See How well we do after year 3
slicer = slicers.HealpixSlicer(nside=64, useCache=False)
metric = metrics.StarDensityMetric(metricName='WhiteDwarfs_rmag<25')
mafMap = maps.StellarDensityMap(startype='wdstars')
plotDict = {'colorMin':0.0001, 'colorMax':0.01, 'logScale':True}
bundle = metricBundles.MetricBundle(metric,slicer,sql, mapsList=[mafMap], plotDict=plotDict)
bundleList.append(bundle)

metric = metrics.StarDensityMetric(rmagLimit=27.5,metricName='WhiteDwarfs_rmag<28')
bundle = metricBundles.MetricBundle(metric,slicer,sql, mapsList=[mafMap], plotDict=plotDict)
bundleList.append(bundle)
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)

bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)



