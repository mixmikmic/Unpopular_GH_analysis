get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from mafContrib import NFollowStacker

# Set the database and query
runName = 'enigma_1189'
opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'FollowUp'
resultsDb = db.ResultsDb(outDir=outDir)

sqlconstraint = 'night < 30'

slicer = slicers.HealpixSlicer(nside=64)
metric = metrics.MeanMetric('nObservatories')
stackerList = [NFollowStacker(minSize=6.5)]
bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, stackerList=stackerList)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

# Change it up to be in alt,az
slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=True)
plotDict = {'rot':(0,90,0)}
bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, stackerList=stackerList, plotDict=plotDict)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

# Let's see what happens with a 24-hour follow-up window
slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=True)
plotDict = {'rot':(0,90,0)}
stackerList = [NFollowStacker(minSize=6.5, timeSteps=np.arange(0,26,2))]
bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, stackerList=stackerList, plotDict=plotDict)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll(closefigs=False)

