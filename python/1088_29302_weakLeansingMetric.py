import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from scipy import stats

class uniformKSTest(metrics.BaseMetric):
    """
    Return the KS-test statistic. Values near zero are good, near 1 is bad.
    """
    def __init__(self, paCol = 'rotSkyPos', modVal=180., metricName='uniformKSTest', units='unitless', **kwargs):
        self.paCol = paCol
        self.modVal = modVal
        super(uniformKSTest, self).__init__(col=paCol, metricName=metricName, units=units, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        angleDist = dataSlice[self.paCol] % self.modVal
        ks_D, pVal = stats.kstest(angleDist, 'uniform')
        return ks_D

class KuiperMetric(metrics.BaseMetric):
    """
    Like the KS test, but for periodic things.
    """
    def __init__(self, col='rotSkyPos', cdf=lambda x:x/(2*np.pi), args=(), period=2*np.pi, **kwargs):
        self.cdf = cdf
        self.args = args
        self.period = period
        assert self.cdf(0) == 0.0
        assert self.cdf(self.period) == 1.0
        super(KuiperMetric, self).__init__(col=col, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        data = np.sort(dataSlice[self.colname] % self.period)
        cdfv = self.cdf(data, *self.args)
        N = len(data)
        D = np.amax(cdfv-np.arange(N)/float(N)) + np.amax((np.arange(N)+1)/float(N)-cdfv)
        return D

opsdb = db.OpsimDatabase('minion_1016_sqlite.db')
outDir = 'temp'
resultsDb = db.ResultsDb(outDir=outDir)

slicer = slicers.HealpixSlicer()
sql = 'filter = "g"'
metric = KuiperMetric()
bundle = metricBundles.MetricBundle(metric, slicer, sql)

bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

bundle.metricValues.max()

slicer = slicers.UniSlicer()
sql = 'fieldID=310 and filter="i"'
metric = metrics.PassMetric('rotSkyPos')

bundle = metricBundles.MetricBundle(metric, slicer, sql)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

ack = plt.hist(np.degrees(bundle.metricValues[0]['rotSkyPos']) % 180.)

ks = uniformKSTest()
print ks.run(bundle.metricValues[0])



