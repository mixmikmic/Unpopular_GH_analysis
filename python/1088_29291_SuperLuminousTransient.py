import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import healpy as hp

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db

# Trying to make a z=2 superluminous 
peaks = {'uPeak':35., 'gPeak':35., 'rPeak':23.3, 'iPeak':22.8, 'zPeak':22.8,'yPeak':22.8}


colors = ['b','g','r','purple','y','magenta','k']
filterNames = ['u','g','r','i','z','y']

peakTime = 90.
transDuration = peakTime+90. # Days
transMetric = metrics.TransientMetric(riseSlope= -1./20., declineSlope=1./20., 
                                      transDuration=transDuration, peakTime=peakTime, surveyDuration=2, 
                                      nFilters=3, nPrePeak=3, nPerLC=2, nPhaseCheck=5, **peaks)


times = np.arange(0.,transDuration*2,1) 
for filterName, color in zip(filterNames,colors):
    filters = np.array([filterName]*times.size)
    lc = transMetric.lightCurve(times % transDuration,filters)
    plt.plot(times,lc, color, label=filterName)
plt.xlabel('time (days)')
plt.ylabel('mags')
plt.ylim([35,18])
plt.legend()

# Pick a slicer
slicer = slicers.HealpixSlicer(nside=64)

summaryMetrics = [metrics.MedianMetric()]

# Set the database and query
runName = 'enigma_1189'
sqlconstraint = 'night < %f' %(365.25*2)

opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'Transients'
resultsDb = db.ResultsDb(outDir=outDir)

bundle = metricBundles.MetricBundle(transMetric, slicer, sqlconstraint, 
                                          runName=runName, summaryMetrics=summaryMetrics)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

bgroup.plotAll(closefigs=False)





