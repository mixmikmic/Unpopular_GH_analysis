get_ipython().magic('matplotlib inline')

# This notebook assumes you are using sims_maf version >= 1.0, and have 'setup sims_maf' in your shell. 
# 

import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import numpy as np
import mafContrib
"""
Run the PhotPrecMetrics 
"""
goodSeeing = 0.7

sqls = [' night < %f' % ( 5.*365.25), ' night < %f and finSeeing < %f'% ( 5.*365.25, goodSeeing)]


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'goodseeing_test'
resultsDb = db.ResultsDb(outDir=outDir)


slicer = slicers.HealpixSlicer(nside=16, lonCol='ditheredRA', latCol='ditheredDec')
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric(), mafContrib.RelRmsMetric()]

bgroupList = []
names = ['All Visits', 'Good Seeing']

for name,sql in zip(names, sqls):
    bundles = {}
    cnt=0
    sed = { 'g':25, 'r': 26, 'i': 25}
    metric1 = mafContrib.SEDSNMetric(metricName='SEDSN', mags=sed)
    metric2 = mafContrib.ThreshSEDSNMetric(metricName='SEDSN', mags=sed)

    bundle1 = metricBundles.MetricBundle(metric1, slicer, sql, summaryMetrics=summaryMetrics)
    bundle2 = metricBundles.MetricBundle(metric2, slicer, sql, summaryMetrics=summaryMetrics)

    bundles={0:bundle1,1:bundle2}

    bgroup = metricBundles.MetricBundleGroup(bundles, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)


for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

if False:
    print 'name, mean PhotPrec, median PhotPrec '
    for bundleGroup in bgroupList:
        for i in range(6):
            print 'Filter %d'%i
            print bundleGroup.bundleDict[i].metric.name,                 bundleGroup.bundleDict[i].summaryValues['Mean'],                 bundleGroup.bundleDict[i].summaryValues['Median'],                bundleGroup.bundleDict[i].summaryValues['RelRms']



