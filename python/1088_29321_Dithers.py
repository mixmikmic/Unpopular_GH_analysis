import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import lsst.sims.maf.db as db
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots

# Grab all the dither stackers from the stock stackers (rather than all stackers)
import inspect
ditherStackerList = []
for name, s in inspect.getmembers(stackers):
    if inspect.isclass(s):
        if 'Dither' in s.__name__:
            ditherStackerList.append(s)

# Print their docstrings
for s in ditherStackerList:
    print '-- ', s.__name__, ' --'
    print s.__doc__
    print '    Generates columns named', s().colsAdded
    print ' '

import numpy as np

def plotDither(ditherStacker, nvisits=1000, addPolygon=True):
    # Set up some 'data' on a single pointing to dither
    fieldIds = np.ones(nvisits, int)
    fieldRA = np.zeros(nvisits, float) + np.radians(10.0)
    fieldDec = np.zeros(nvisits, float) 
    night = np.arange(0, nvisits/2.0, 0.5)
    night = np.floor(night) 
    simdata = np.core.records.fromarrays([fieldIds, fieldRA, fieldDec, night], 
                                         names = ['fieldID', 'fieldRA', 'fieldDec', 'night'])
    # Apply the stacker. 
    simdata = ditherStacker.run(simdata)
    
    fig = plt.figure()
    plt.axis('equal')
    # Draw a point for the center of the FOV.
    x = np.degrees(simdata['fieldRA'][0])
    y = np.degrees(simdata['fieldDec'][0])
    plt.plot(x, y, 'g+')
    # Draw a circle approximately the size of the FOV.
    stepsize = np.pi/50.
    theta = np.arange(0, np.pi*2.+stepsize, stepsize)
    radius = 1.75
    plt.plot(radius*np.cos(theta)+x, radius*np.sin(theta)+y, 'g-')
    # Add the inscribed hexagon
    nside = 6
    a = np.arange(0, nside)
    xCoords = np.sin(2*np.pi/float(nside)*a + np.pi/2.0)*radius + x
    yCoords = np.cos(2*np.pi/float(nside)*a + np.pi/2.0)*radius + y
    xCoords = np.concatenate([xCoords, np.array([xCoords[0]])])
    yCoords = np.concatenate([yCoords, np.array([yCoords[0]])])
    plt.plot(xCoords, yCoords, 'b-')
    # Draw the dithered pointings.
    x = np.degrees(simdata[s.colsAdded[0]])
    y = np.degrees(simdata[s.colsAdded[1]])
    plt.plot(x, y, 'k-', alpha=0.2)
    plt.plot(x, y, 'r.')
    plt.title(s.__class__.__name__)

for ditherStacker in ditherStackerList:
    s = ditherStacker()
    plotDither(s)

s = stackers.SpiralDitherFieldVisitStacker()
plotDither(s, nvisits=30)
s = stackers.SpiralDitherFieldNightStacker()
plotDither(s, nvisits=30)

s = stackers.RandomDitherFieldVisitStacker(maxDither=1.75, inHex=False)
plotDither(s)
s = stackers.RandomDitherFieldVisitStacker(maxDither=1.75, inHex=True)  #inHex is true by default
plotDither(s)
s = stackers.RandomDitherFieldVisitStacker(maxDither=0.5)
plotDither(s)

s = stackers.RandomDitherFieldVisitStacker(randomSeed=253)
plotDither(s, nvisits=200)
s = stackers.RandomDitherFieldVisitStacker(randomSeed=100)
plotDither(s, nvisits=200)

s = stackers.SpiralDitherFieldVisitStacker(nCoils=3)
plotDither(s)
s = stackers.SpiralDitherFieldVisitStacker(nCoils=6)
plotDither(s)
s = stackers.SpiralDitherFieldVisitStacker(nCoils=6, inHex=False)
plotDither(s)

nside = 128
metric = metrics.CountMetric('expMJD')
slicer0 = slicers.HealpixSlicer(lonCol='fieldRA', latCol='fieldDec', nside=nside)  
#sqlconstraint = 'filter="r" and night<730' 
sqlconstraint = 'filter="r"'

myBundles = {}
myBundles['no dither'] = metricBundles.MetricBundle(metric, slicer0, sqlconstraint, 
                                                    runName='enigma_1189', metadata='no dither')

# ditheredRA and ditheredDec correspond to the stock opsim dither pattern
slicer1 = slicers.HealpixSlicer(lonCol='ditheredRA', latCol='ditheredDec', nside=nside)
myBundles['hex dither'] = metricBundles.MetricBundle(metric, slicer1, sqlconstraint, runName='enigma_1189', 
                                                    metadata = 'hex dither')

slicer2 = slicers.HealpixSlicer(lonCol='randomDitherFieldVisitRa', latCol='randomDitherFieldVisitDec', nside=nside)
myBundles['random dither'] = metricBundles.MetricBundle(metric, slicer2, sqlconstraint, runName='enigma_1189',
                                                       metadata='random dither')

stackerList = [stackers.SpiralDitherFieldNightStacker(nCoils=7)]
slicer3 = slicers.HealpixSlicer(lonCol='spiralDitherFieldNightRa', latCol='spiralDitherFieldNightDec', nside=nside)
myBundles['spiral dither'] = metricBundles.MetricBundle(metric, slicer3, sqlconstraint, 
                                                       stackerList=stackerList, runName='enigma_1189',
                                                       metadata='spiral dither')

opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'dither_test'
resultsDb = db.ResultsDb(outDir=outDir)
bgroup = metricBundles.MetricBundleGroup(myBundles, opsdb, outDir=outDir, resultsDb=resultsDb)

#import os
#for b in bgroup.bundleDict.itervalues():
#    filename = os.path.join(outDir, b.fileRoot) + '.npz'
#    b.read(filename)

bgroup.runAll()

ph = plots.PlotHandler(outDir=outDir, resultsDb=resultsDb)

for mB in myBundles.itervalues():
    plotDict = {'xMin':0, 'xMax':300, 'colorMin':0, 'colorMax':300}
    mB.setPlotDict(plotDict)
    mB.plot(plotFunc=plots.HealpixSkyMap, plotHandler=ph)

ph.setMetricBundles(myBundles)
# We must set a series of plotDicts: one per metricBundle. 
#  because we don't explicitly set the colors, they will be set randomly. 
plotDict = {'binsize':1, 'xMin':0, 'xMax':350}
ph.plot(plots.HealpixHistogram(), plotDicts=plotDict)

# Plot some close-ups.
import healpy as hp
for mB in myBundles.itervalues():
    hp.cartview(mB.metricValues, lonra=[70, 100], latra=[-30, -0], min=150., max=300., 
            flip='astro', title=mB.metadata)

summaryMetrics = [metrics.TotalPowerMetric()]
for mB in myBundles.itervalues():
    mB.setSummaryMetrics(summaryMetrics)
    mB.computeSummaryStats()
    plotDict = {'label':'%s : %g' %(mB.metadata, mB.summaryValues['TotalPower'])}
    mB.setPlotDict(plotDict)

ph.plot(plots.HealpixPowerSpectrum(), plotDicts={'legendloc':(1, 0.3)})



