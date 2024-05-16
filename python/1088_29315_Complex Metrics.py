import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils

opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')

propinfo, proptags = opsdb.fetchPropInfo()
reqvisits = opsdb.fetchRequestedNvisits(proptags['WFD'])
print reqvisits

completeness_metric = metrics.CompletenessMetric(u=reqvisits['u'], g=reqvisits['g'], r=reqvisits['r'], 
                                          i=reqvisits['i'], z=reqvisits['z'], y=reqvisits['y'])
slicer = slicers.OpsimFieldSlicer()
sqlconstraint = utils.createSQLWhere('WFD', proptags)
summaryMetric = metrics.TableFractionMetric()
plotDict = {'xMin':0, 'xMax':1.2, 'colorMin':0, 'colorMax':1.2, 'binsize':0.025}

completeness = metricBundles.MetricBundle(metric=completeness_metric, slicer=slicer, 
                                          sqlconstraint=sqlconstraint, runName='enigma_1189', 
                                          summaryMetrics=summaryMetric, plotDict=plotDict)

bdict = {'completeness':completeness}

outDir = 'completeness_test'
resultsDb = db.ResultsDb(outDir=outDir)
bg = metricBundles.MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)

bg.runAll()

bg.plotAll(closefigs=False)

for b in bdict.itervalues():
    print b.metric.name, b.summaryMetrics, b.summaryValues

