get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
from lsst.sims.maf.plots import PlotHandler
import healpy as hp

# Set the database and query
database = 'enigma_1189_sqlite.db'
#sqlWhere = 'filter = "r" and night < 400'
opsdb = db.OpsimDatabase(database)
outDir = 'Camera'
resultsDb = db.ResultsDb(outDir=outDir)
nside=512

rafts = ['R:0,1', 'R:0,2', 'R:0,3',
         'R:1,0', 'R:1,1', 'R:1,2', 'R:1,3', 'R:1,4',
         'R:2,0', 'R:2,1', 'R:2,2', 'R:2,3', 'R:2,4',
         'R:3,0', 'R:3,1', 'R:3,2', 'R:3,3', 'R:3,4',
         'R:4,1', 'R:4,2', 'R:4,3',
        ]
chips = ['S:0,0', 'S:0,1', 'S:0,2',
        'S:1,0', 'S:1,1', 'S:1,2',
        'S:2,0', 'S:2,1', 'S:2,2']
allChips =[]
for raft in rafts:
    for chip in chips:
        allChips.append(raft+' '+chip)

sqlWhere = 'filter = "r" and expMJD < 49547.36 and fieldRA < %f and fieldDec > %f and fieldDec < 0' % (np.radians(15.), np.radians(-15.))
metric = metrics.Coaddm5Metric()
slicer = slicers.HealpixSlicer(nside=nside, useCamera=True, chipNames=allChips)

bundle = metricBundles.MetricBundle(metric,slicer,sqlWhere)
bg = metricBundles.MetricBundleGroup({0:bundle},opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()
bg.plotAll(closefigs=False)
hp.gnomview(bundle.metricValues, xsize=800,ysize=800, rot=(0,0,0))

# Now let's use every-other chip
halfChips = []
for raft in rafts[0::2]:
    for chip in chips:
        halfChips.append(raft+' '+chip)

slicer = slicers.HealpixSlicer(nside=nside, useCamera=True, chipNames=halfChips)

bundle = metricBundles.MetricBundle(metric,slicer,sqlWhere)
bg = metricBundles.MetricBundleGroup({0:bundle},opsdb, outDir=outDir, resultsDb=resultsDb)
bg.runAll()
bg.plotAll(closefigs=False)
hp.gnomview(bundle.metricValues, xsize=800,ysize=800, rot=(0,0,0))







