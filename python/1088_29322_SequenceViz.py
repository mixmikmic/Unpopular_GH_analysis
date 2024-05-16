import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db



runName = 'enigma_1189'

opsdb = db.OpsimDatabase(runName + '_sqlite.db')
outDir = 'Transients'
resultsDb = db.ResultsDb(outDir=outDir)

metric=metrics.PassMetric(cols=['expMJD', 'fiveSigmaDepth', 'filter'])
ra = [0.]
dec = [np.radians(-30.)]
slicer = slicers.UserPointsSlicer(ra=ra,dec=dec)
sqlconstraint = 'night < 365'

bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, runName=runName)
bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

filters = np.unique(bundle.metricValues[0].compressed()['filter'])
colors = {'u':'b','g':'g','r':'r','i':'purple',"z":'y',"y":'magenta'}

mv = bundle.metricValues[0].compressed()
for filterName in filters:
    good = np.where(mv['filter'] == filterName)
    plt.scatter(mv['expMJD'][good]-mv['expMJD'].min(), mv['fiveSigmaDepth'][good], 
                c=colors[filterName], label=filterName)
plt.xlabel('Day')
plt.ylabel('5$\sigma$ depth (mags)')
plt.xlim([0,100])
plt.legend(scatterpoints=1)





