import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles

filterName = 'r'
years = [1.,5.,10.]
nights = np.array(years)*365.25
sqls = ['filter = "%s" and night < %f' %(filterName, night) for night in nights]
print sqls

# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'depths_test'
resultsDb = db.ResultsDb(outDir=outDir)

slicer = slicers.HealpixSlicer()
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric()]
metric = metrics.Coaddm5Metric()
bgroupList = []
for year,sql in zip(years,sqls):
    bundle = metricBundles.MetricBundle(metric, slicer, sql, summaryMetrics=summaryMetrics)
    bundle.plotDict['label'] = '%i' % year
    bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)
    

for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

print 'year, mean depth, median depth'
for year,bundleGroup in zip(years,bgroupList):
    print year, bundleGroup.bundleDict[0].summaryValues['Mean'], bundleGroup.bundleDict[0].summaryValues['Median']

import healpy as hp
for year,bundleGroup in zip(years,bgroupList):
    hp.gnomview(bundleGroup.bundleDict[0].metricValues, rot=(0,-30), title='year %i'%year, 
                min=25.5, max=27.8, xsize=500,ysize=500)

