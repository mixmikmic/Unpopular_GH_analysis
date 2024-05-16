# For reference, here are the parameters used to simulate the transients:
#peaks = {'uPeak':11, 'gPeak':9, 'rPeak':8, 'iPeak':7, 'zPeak':6,'yPeak':6}
#colors = ['b','g','r','purple','y','magenta','k']
#filterNames = ['u','g','r','i','z','y']
## Timing parameters of the outbursts
#riseSlope = -2.4
#declineSlope = 0.05  # following Ofek et al. 2013
#transDuration = 80.
#peakTime = 20.

# relevant parameter for the TransientMetric:
# nPhaseCheck=20

# Additionally, all filters were used (passed as **peaks to the TransientMetric).

get_ipython().magic('matplotlib inline')

import numpy as np
import time

# Some colormaps we might use
import matplotlib.cm as cm

# Capability to load previously-computed metrics, examine them
import lsst.sims.maf.metricBundles as mb

# plotting (to help assess the results)
import lsst.sims.maf.plots as plots

# The example CountMetric provided by Mike Lund seems to have the column indices for coords
# hardcoded (which breaks the examples I try on my setup). This version finds the co-ordinates by 
# name instead. First the imports we need:
# import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from mafContrib import starcount 

class AsCountMetric(BaseMetric):

    """
    WIC - Lightly modified copy of Mike Lund's example StarCounts metric in sims_maf_contrib. 
    Accepts the RA, DEC column names as keyword arguments. Docstring from the original:
    
    Find the number of stars in a given field between distNear and distFar in parsecs. 
    Field centers are read from columns raCol and decCol.
    """
    
    def __init__(self,**kwargs):
        
        self.distNear=kwargs.pop('distNear', 100)
        self.distFar=kwargs.pop('distFar', 1000)
        self.raCol=kwargs.pop('raCol', 'ra')
        self.decCol=kwargs.pop('decCol', 'dec')
        super(AsCountMetric, self).__init__(col=[], **kwargs)
        
    def run(self, dataSlice, slicePoint=None):
        sliceRA = np.degrees(slicePoint[self.raCol])
        sliceDEC = np.degrees(slicePoint[self.decCol])
        return starcount.starcount(sliceRA, sliceDEC, self.distNear, self.distFar)

distNear=10.
distFar = 8.0e4  # Get most of the plane but not the magellanic clouds 

import lsst.sims.maf.slicers as slicers

import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db

slicer = slicers.HealpixSlicer(nside=64)

metricCount=AsCountMetric(distNear=distNear, distFar=distFar)
metricList = [metricCount]

runName1092 = 'ops2_1092'
sqlconstraintCount = 'filter = "r" & night < 1000'  # Assume everywhere visited once in three days...
bDict1092={}
for i,metric in enumerate(metricList):
    bDict1092[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraintCount, 
                                          runName=runName1092)
opsdb1092 = db.OpsimDatabase(runName1092 + '_sqlite.db')
outDir1092 = 'TestCountOnly1092'
resultsDb1092 = db.ResultsDb(outDir=outDir1092)

tStart = time.time()
bgroup1092 = metricBundles.MetricBundleGroup(bDict1092, opsdb1092, outDir=outDir1092,                                              resultsDb=resultsDb1092)
bgroup1092.runAll()
tPost1092 = time.time()
print "Time spent Counting 1092: %.3e seconds" % (tPost1092 - tStart)

# Ensure the output file actually got written...
get_ipython().system(' ls -l ./TestCountOnly1092/*npz')

# We will need the same counts information for enigma if we want to normalize by 
# total counts in the survey area. So let's run the above for enigma_1189 as well.
runName1189 = 'enigma_1189'
bDict1189={}
for i,metric in enumerate(metricList):
    bDict1189[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraintCount, 
                                          runName=runName1189)
opsdb1189 = db.OpsimDatabase(runName1189 + '_sqlite.db')
outDir1189 = 'TestCountOnly1189'
resultsDb1189 = db.ResultsDb(outDir=outDir1189)

tStart = time.time()
bgroup1189 = metricBundles.MetricBundleGroup(bDict1189, opsdb1189, outDir=outDir1189,                                              resultsDb=resultsDb1189)
bgroup1189.runAll()
tPost1189 = time.time()
print "Time spent Counting 1189: %.3e seconds" % (tPost1189 - tStart)

get_ipython().system(' ls ./TestCountOnly1189/enigma_1189_AsCount_r_HEAL.npz')

pathCount='./TestCountOnly1092/ops2_1092_AsCount_r_HEAL.npz'
pathTransient='Transients1092Like2010mc/ops2_1092_Alert_sawtooth_HEAL.npz'

#Initialize then load
bundleCount = mb.createEmptyMetricBundle()
bundleTrans = mb.createEmptyMetricBundle()

bundleCount.read(pathCount)
bundleTrans.read(pathTransient)

# Set a mask for the BAD values of the transient metric
bTrans = (np.isnan(bundleTrans.metricValues)) | (bundleTrans.metricValues <= 0.)
bundleTrans.metricValues.mask[bTrans] = True

# Read in the stellar density for 1189 so that we can compare the total NStars...
pathCount1189='./TestCountOnly1189/enigma_1189_AsCount_r_HEAL.npz'
bundleCount1189 = mb.createEmptyMetricBundle()
bundleCount1189.read(pathCount1189)

# Do the comparison
nTot1092 = np.sum(bundleCount.metricValues)
nTot1189 = np.sum(bundleCount1189.metricValues)
print "Total NStars - ops2_1092: %.3e - enigma_1189 %.3e" % (nTot1092, nTot1189)

bundleProc = mb.createEmptyMetricBundle()
bundleProc.read(pathTransient)

# Set the mask
bundleProc.metricValues.mask[bTrans] = True

# Multiply the two together, normalise by the total starcounts over the survey
bundleProc.metricValues = (bundleCount.metricValues * bundleTrans.metricValues) 
bundleProc.metricValues /= np.sum(bundleCount.metricValues)

bundleProc.metric.name = '(sawtooth alert) x (counts) / NStars_total'

FoM1092 = np.sum(bundleProc.metricValues)
print "FoM 1092: %.2e" % (FoM1092)

pathCount1189='./TestCountOnly1189/enigma_1189_AsCount_r_HEAL.npz'
pathTrans1189='./Transients1189Like2010mc/enigma_1189_Alert_sawtooth_HEAL.npz'
bundleCount1189 = mb.createEmptyMetricBundle()
bundleTrans1189 = mb.createEmptyMetricBundle()

bundleCount1189.read(pathCount1189)
bundleTrans1189.read(pathTrans1189)
bTrans1189 = (np.isnan(bundleTrans1189.metricValues)) | (bundleTrans1189.metricValues <= 0.)
bundleTrans1189.metricValues.mask[bTrans1189] = True

# Load 1189-like metric bundle and replace its values with processed values
bundleProc1189 = mb.createEmptyMetricBundle()
bundleProc1189.read(pathTrans1189)
bundleProc1189.metricValues.mask[bTrans1189] = True

bundleProc1189.metricValues = (bundleCount1189.metricValues * bundleTrans1189.metricValues) 
bundleProc1189.metricValues /= np.sum(bundleCount1189.metricValues)
bundleProc1189.metric.name = '(sawtooth alert) x (counts) / NStars_total'

FoM1189 = np.sum(bundleProc1189.metricValues)
print FoM1189

# Print the sum total of our f.o.m. for each run
print "FOM for ops2_1092: %.3f" % (FoM1092)
print "FOM for enigma_1189: %.3f" % (FoM1189)

# Same plot information as before:
plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
plotDictProc={'logScale':True, 'cmap':cm.cubehelix_r}
bundleProc.setPlotDict(plotDictProc)
bundleProc.setPlotFuncs(plotFuncs)

plotDictProc={'logScale':True, 'cmap':cm.cubehelix_r}
bundleProc1189.setPlotDict(plotDictProc)
bundleProc1189.setPlotFuncs(plotFuncs)



bundleProc.plot(savefig=True)
bundleProc1189.plot(savefig=True)



# Plot just the spatial map and the histogram for the two. Use different colormaps for each.
#plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
bundleTrans.setPlotFuncs(plotFuncs)
bundleCount.setPlotFuncs(plotFuncs)

# Use a different colormap for each so we can tell them apart easily...
plotDictCount={'logScale':True, 'cmap':cm.gray_r}
plotDictTrans={'logScale':False, 'cmap':cm.RdBu_r}
bundleCount.setPlotDict(plotDictCount)
bundleTrans.setPlotDict(plotDictTrans)

plotDictCount={'logScale':True, 'cmap':cm.gray_r}
plotDictTrans={'logScale':False, 'cmap':cm.RdBu_r}
bundleCount1189.setPlotDict(plotDictCount)
bundleTrans1189.setPlotDict(plotDictTrans)
bundleTrans1189.setPlotFuncs(plotFuncs)
bundleCount1189.setPlotFuncs(plotFuncs)

bundleCount.plot()
bundleTrans.plot()

bundleCount1189.plot()
bundleTrans1189.plot()



