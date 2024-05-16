get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles

class NnightsWColor(metrics.BaseMetric):
    """See how many nights a spot is observed at least twice in one filter and once in another.
    
    Parameters
    ----------
    n_in_one : int (2)
       The number of observations in a single filter to require
    n_filts : int (2)
       The number of unique filters to demand
    filters : list 
       The list of acceptable filters
    
    """
    def __init__(self, metricName='', mjdCol='expMJD',
                 filterCol='filter', nightCol='night', n_in_one=2, n_filts=2, 
                 filters=None, **kwargs):
        if filters is None:
            self.filters = ['u', 'g', 'r', 'i', 'z', 'y']
        else:
            self.filters = filters
        self.mjdCol = mjdCol
        self.nightCol = nightCol
        self.filterCol = filterCol
        self.n_in_one = n_in_one
        self.n_filts = n_filts
        super(NnightsWColor, self).__init__(col=[self.mjdCol, self.nightCol,
                                                 self.filterCol],
                                            units='# Nights',
                                            metricName=metricName,
                                            **kwargs)
        
    def run(self,  dataSlice, slicePoint=None):
        
        night_bins = np.arange(dataSlice[self.nightCol].min()-.5, dataSlice[self.nightCol].max()+2.5, 1)
        all_obs = np.zeros((night_bins.size-1, len(self.filters)))
        for i, filtername in enumerate(self.filters):
            good = np.where(dataSlice[self.filterCol] == filtername)
            hist, edges = np.histogram(dataSlice[self.nightCol][good], bins=night_bins)
            all_obs[:,i] += hist
        # max number of observations in a single filter per night
        max_collapse = np.max(all_obs, axis=1)
        all_obs[np.where(all_obs > 1)] = 1
        # number of unique filters per night
        n_filt = np.sum(all_obs, axis=1)
        good = np.where((max_collapse >= self.n_in_one) & (n_filt >= self.n_filts))[0]
        return np.size(good)

runName = 'minion_1016'
opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'TransientsUPS'
resultsDb = db.ResultsDb(outDir=outDir)

metric = NnightsWColor()
sql = ''
slicer = slicers.HealpixSlicer()
bundle = metricBundles.MetricBundle(metric, slicer, sql, runName=runName)

bundleList = [bundle]
bundleDict = metricBundles.makeBundlesDictFromList(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

bgroup.plotAll(closefigs=False)



