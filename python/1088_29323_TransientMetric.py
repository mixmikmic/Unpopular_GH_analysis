import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import healpy as hp

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db

# Let's look at how the transient metric makes light curves
peaks = {'uPeak':18, 'gPeak':19, 'rPeak':20, 'iPeak':21, 'zPeak':22,'yPeak':23}
colors = ['b','g','r','purple','y','magenta','k']
filterNames = ['u','g','r','i','z','y']

transDuration = 60. # Days
transMetric = metrics.TransientMetric(riseSlope=-2., declineSlope=0.25, 
                                      transDuration=transDuration, peakTime=5., **peaks)


times = np.arange(0.,121,1) 
for filterName, color in zip(filterNames,colors):
    filters = np.array([filterName]*times.size)
    lc = transMetric.lightCurve(times % transDuration,filters)
    plt.plot(times,lc, color, label=filterName)
plt.xlabel('time (days)')
plt.ylabel('mags')
plt.ylim([35,18])
plt.legend()

# Modify the slopes and duration a bit
transDuration = 10.
transMetric = metrics.TransientMetric(riseSlope=-1., declineSlope=1, transDuration=transDuration, 
                                 peakTime=5., **peaks)


times = np.arange(0.,121,1) 
for filterName, color in zip(filterNames,colors):
    filters = np.array([filterName]*times.size)
    lc = transMetric.lightCurve(times % transDuration,filters)
    plt.plot(times,lc, color, label=filterName)
plt.xlabel('time (days)')
plt.ylabel('mags')
plt.ylim([30,18])
plt.legend()

# Pick a slicer
slicer = slicers.HealpixSlicer(nside=64)

summaryMetrics = [metrics.MedianMetric()]
# Configure some metrics
metricList = []
# What fraction of 60-day, r=20 mag flat transients are detected at least once?
metric = metrics.TransientMetric(riseSlope=0., declineSlope=0., transDuration=60., 
                                 peakTime=5., rPeak=20., metricName='Alert')
metricList.append(metric)
# Now make the light curve shape a little more realistic. 
metric = metrics.TransientMetric(riseSlope=-2., declineSlope=0.25, transDuration=60., 
                                 peakTime=5., rPeak=20., metricName='Alert, shaped LC')
metricList.append(metric)
#  Demand 2 points before tmax before counting the LC as detected
metric = metrics.TransientMetric(riseSlope=-2., declineSlope=0.25, transDuration=60., 
                                 peakTime=5., rPeak=20., nPrePeak=2, metricName='Detected on rise')
metricList.append(metric)

# Set the database and query
runName = 'enigma_1189'
sqlconstraint = 'filter = "r"'
bDict={}
for i,metric in enumerate(metricList):
    bDict[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, 
                                          runName=runName, summaryMetrics=summaryMetrics)

opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'Transients'
resultsDb = db.ResultsDb(outDir=outDir)

bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

bgroup.plotAll(closefigs=False)

# Compute and print summary metrics
for key in bDict:
    bDict[key].computeSummaryStats(resultsDb=resultsDb)
    print bDict[key].metric.name, bDict[key].summaryValues

# Update to use all the observations, not just the r-band
bDict={}
sqlconstraint = ''
for i,metric in enumerate(metricList):
    bDict[i] = metricBundles.MetricBundle(metric, slicer, sqlconstraint, summaryMetrics=summaryMetrics)

bgroup = metricBundles.MetricBundleGroup(bDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)
for key in bDict:
    bDict[key].computeSummaryStats(resultsDb=resultsDb)
    print bDict[key].metric.name, bDict[key].summaryValues

bDict[0].metricValues





