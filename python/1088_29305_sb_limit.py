import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import math

# This is needed to avoid an error when a metric is redefined
from lsst.sims.maf.metrics import BaseMetric
try:
    del metrics.BaseMetric.registry['__main__.SB']
except KeyError:
    pass

class SB(BaseMetric):
    """Calculate the SB at this gridpoint."""
    def __init__(self, m5Col = 'fiveSigmaDepth', metricName='SB', **kwargs):
        """Instantiate metric.

        m5col = the column name of the individual visit m5 data."""
        super(SB, self).__init__(col=m5Col, metricName=metricName, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        seeing = 0.7
        return 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.colname])) * (math.pi*seeing**2))

filterName = 'r'
years = [1, 2, 3, 5, 10]
nights = np.array(years)*365.25
sqls = ['filter = "%s" and night < %f' %(filterName, night) for night in nights]
print sqls

# Set up the database connection
dbdir = '/Users/loveday/sw/lsst/enigma_1189/'
opsdb = db.OpsimDatabase(database = os.path.join(dbdir, 'enigma_1189_sqlite.db'))
outDir = 'GoodSeeing'
resultsDb = db.ResultsDb(outDir=outDir)

# opsdb.tables['Summary'].columns

slicer = slicers.HealpixSlicer()
summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric()]
metric = SB()
bgroupList = []
for year,sql in zip(years,sqls):
    bundle = metricBundles.MetricBundle(metric, slicer, sql, summaryMetrics=summaryMetrics)
    bundle.plotDict['label'] = '%i' % year
    bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroupList.append(bgroup)

for bgroup in bgroupList:
    bgroup.runAll()
    bgroup.plotAll(closefigs=False)

mean_depth = []
median_depth = []
print 'year, mean depth, median depth'
for year,bundleGroup in zip(years,bgroupList):
    mean_depth.append(bundleGroup.bundleDict[0].summaryValues['Mean'])
    median_depth.append(bundleGroup.bundleDict[0].summaryValues['Median'])
    print (year, bundleGroup.bundleDict[0].summaryValues['Mean'], 
           bundleGroup.bundleDict[0].summaryValues['Median'])

# Plot SB limits as fn of time
plt.clf()
plt.plot(years, mean_depth, label='mean')
plt.plot(years, median_depth, label='median')
plt.plot((years[0], years[-1]), (26, 26), ':', label='~1:10 mass ratio')
plt.xlabel('Time (years)')
plt.ylabel(r'SB (r mag arcsec$^{-2}$)')
plt.legend(loc=4)
plt.show()






