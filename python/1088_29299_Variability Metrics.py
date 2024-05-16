import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# Import MAF modules.
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
from lsst.sims.maf.metricBundles import MetricBundle, MetricBundleGroup
# Import the contributed metrics and stackers 
import mafContrib

runName = 'enigma_1189'
database =  runName + '_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'variability_test'

phaseMetric = metrics.PhaseGapMetric(nPeriods=20, periodMin=0.2, periodMax=3.5)
periodMetric = mafContrib.PeriodDeviationMetric(nPeriods=2, periodMin=2, periodMax=3.5)
phaseslicer = slicers.HealpixSlicer(nside=64, lonCol='ditheredRA', latCol='ditheredDec')
periodslicer = slicers.HealpixSlicer(nside=32, lonCol='ditheredRA', latCol='ditheredDec')

summaryMetrics = [metrics.MinMetric(), metrics.MaxMetric(), metrics.MeanMetric()]

sqlconstraint = 'night<365 and (filter="r" or filter="i")'
phaseBundle = MetricBundle(phaseMetric, phaseslicer, sqlconstraint=sqlconstraint, 
                           runName=runName, summaryMetrics=summaryMetrics)
periodBundle = MetricBundle(periodMetric, periodslicer, sqlconstraint=sqlconstraint, 
                            runName=runName, summaryMetrics=summaryMetrics)

bdict = {'Phase':phaseBundle, 'Period':periodBundle}

resultsDb = db.ResultsDb(outDir=outDir)
bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)

bgroup.runAll()

bgroup.plotAll(closefigs=False)

def count_number(x, y, xbinsize=None, ybinsize=None, nxbins=None, nybins=None):
    # Set up grid for contour/density plot.
    xmin = min(x)
    ymin = min(y)
    if (xbinsize!=None) & (ybinsize!=None):
        xbins = np.arange(xmin, max(x), xbinsize)
        ybins = np.arange(ymin, max(y), ybinsize)
        nxbins = xbins.shape[0]
        nybins = ybins.shape[0]
    elif (nxbins!=None) & (nybins!=None):
        xbinsize = (max(x) - xmin)/float(nxbins)
        ybinsize = (max(y) - ymin)/float(nybins)
        xbins = np.arange(xmin, max(x), xbinsize)
        ybins = np.arange(ymin, max(y), ybinsize)
        nxbins = xbins.shape[0]
        nybins = ybins.shape[0]
    else:
        raise Exception("Must specify both of either xbinsize/ybinsize or nxbins/nybins")
    counts = np.zeros((nybins, nxbins), dtype='int')
    # Assign each data point (x/y) to a bin.
    for i in range(len(x)):
        xidx = min(int((x[i] - xmin)/xbinsize), nxbins-1)
        yidx = min(int((y[i] - ymin)/ybinsize), nybins-1)
        counts[yidx][xidx] += 1
    # Create 2D x/y arrays, to match 2D counts array.
    xi, yi = np.meshgrid(xbins, ybins)
    return xi, yi, counts

nperiods = len(phaseBundle.metricValues[-1:][0]['periods'])
periods = []
phaseGaps = []
for mval in phaseBundle.metricValues.compressed():
    for p, pGap in zip(mval['periods'], mval['maxGaps']):
            periods.append(p)
            phaseGaps.append(pGap)

periods = np.array(periods, 'float')
phaseGaps = np.array(phaseGaps, 'float')
timeGaps = phaseGaps * periods

periodi, phasegapi, counts = count_number(periods, phaseGaps, nxbins=100, nybins=100)
plt.figure()
levels = np.log10(np.arange(0.1, 200, 1))
plt.contourf(periodi, phasegapi, np.log10(counts), levels, extend='max')
cbar = plt.colorbar()
cbar.set_label('logN')
plt.xlabel('Period (days)')
plt.ylabel('Largest Phase gap')

nperiods = len(periodBundle.metricValues[-1:][0]['periods'])
periods = []
periodsdev = []
for mval in periodBundle.metricValues.compressed():
    for p, pdev in zip(mval['periods'], mval['periodsdev']):
            periods.append(p)
            periodsdev.append(pdev)

periods = np.array(periods, 'float')
periodsdev = np.array(periodsdev, 'float')
fitperiods = periodsdev * periods + periods

periodi, periodsdevi, counts = count_number(periods, periodsdev, nxbins=100, nybins=100)
plt.figure()
levels = np.arange(0, 15, .1)
#levels = np.log10(levels)
#counts = np.log10(counts)
plt.contourf(periodi, periodsdevi, counts, levels, extend='max')
cbar = plt.colorbar()
cbar.set_label('logN')
plt.xlabel('Period (days)')
plt.ylabel('Period deviation')


plt.figure()
plt.plot(periods, fitperiods, 'k.')
plt.xlabel('True Period (days)')
plt.ylabel('Fit Period (days)')

periodi, fitperiodsi, counts = count_number(periods, fitperiods, nxbins=100, nybins=100)
plt.figure()
#counts = np.log10(counts)
plt.contourf(periodi, fitperiodsi, counts, levels, extend='max')
cbar = plt.colorbar()
cbar.set_label('logN')
plt.xlabel('True Period (days)')
plt.ylabel('Fit period (days)')

# Check how well we can recover a 3-day period variable
periodMetric = mafContrib.PeriodDeviationMetric(nPeriods=2, periodMin=2, periodMax=3.5, periodCheck=3.)
periodslicer = slicers.HealpixSlicer(nside=32, lonCol='ditheredRA', latCol='ditheredDec')
sqlconstraint = 'night<365 and (filter="r" or filter="i")'
periodBundle = MetricBundle(periodMetric, periodslicer, sqlconstraint=sqlconstraint, 
                            runName=runName)
bdict = {'Period':periodBundle}
bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)



