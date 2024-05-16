import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles

# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'astrometry_test'
resultsDb = db.ResultsDb(outDir=outDir)

sql = 'night < %i' % (365.25*3) # See How well we do after year 3
slicer = slicers.HealpixSlicer(nside=64)
metricList = []
metricList.append(metrics.ParallaxMetric())
metricList.append(metrics.ParallaxMetric(metricName='properMotion Normed', normalize=True))
metricList.append(metrics.ProperMotionMetric())
metricList.append(metrics.ProperMotionMetric(metricName='properMotion Normed', normalize=True))

summaryList = [metrics.MedianMetric()]

bundleList = []
for metric in metricList:
    bundleList.append(metricBundles.MetricBundle(metric,slicer,sql, summaryMetrics=summaryList))

bundleDict = metricBundles.makeBundleDict(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

bgroup.plotAll(closefigs=False)

rmags = {'faint':25, 'bright':18}
specTypes = ['B', 'K']
metricList = []
for mag in rmags:
    for specType in specTypes:
        metricList.append(metrics.ParallaxMetric(rmag=rmags[mag], SedTemplate=specType, 
                                                 metricName='parallax_'+mag+'_'+specType))
        metricList.append(metrics.ProperMotionMetric(rmag=rmags[mag], SedTemplate=specType, 
                                                     metricName='properMotion'+mag+'_'+specType))

bundlesSpec = []
for metric in metricList:
    bundlesSpec.append(metricBundles.MetricBundle(metric,slicer,sql, summaryMetrics=summaryList))
bundlesSpec = metricBundles.makeBundleDict(bundlesSpec)

bgroup = metricBundles.MetricBundleGroup(bundlesSpec, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

bgroup.plotAll(closefigs=False)

print 'Flat SED:'
for bundle in bundles.values():
    print bundle.metric.name, bundle.summaryValues
print 'B and K stars:'
for bundle in bundlesSpec.values():
    print bundle.metric.name, bundle.summaryValues



