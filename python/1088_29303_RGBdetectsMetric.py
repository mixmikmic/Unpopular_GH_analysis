import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles

filterName1 = 'g'
filterName2 = 'i'
years=[1.,5.,10.]
#nights = np.array(years)*365.25
goodSeeing = 0.7
sqls = ['filter = "%s" and night < %f and finSeeing < %f'% (filterName1, 5.*365.25, goodSeeing),
        'filter = "%s" and night < %f and finSeeing < %f'% (filterName2, 5.*365.25, goodSeeing)]
print sqls

# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'goodseeing_test'
resultsDb = db.ResultsDb(outDir=outDir)

slicer = slicers.HealpixSlicer(lonCol='ditheredRA',latCol='ditheredDec')
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric()]

bgroupList = []
names = ['All Visits', 'Good Seeing']
for name,sql in zip(names, sqls):
    metric = metrics.Coaddm5Metric(metricName=name)
    bundle = metricBundles.MetricBundle(metric, slicer, sql, summaryMetrics=summaryMetrics)
    bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)

for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

print 'name, mean depth, median depth'
for bundleGroup in bgroupList:
    print bundleGroup.bundleDict[0].metric.name, bundleGroup.bundleDict[0].summaryValues['Mean'], bundleGroup.bundleDict[0].summaryValues['Median']

b = bundleGroup.bundleDict[0]

b.metricValues.data[10000]



