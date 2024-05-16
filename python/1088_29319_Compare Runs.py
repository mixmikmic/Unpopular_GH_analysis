# import the modules needed.
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from lsst.sims.maf.db import ResultsDb

rootDir = '.'
rundb = {}
rundb['enigma_1189'] = ResultsDb(database=os.path.join(rootDir, 'enigma_1189_scheduler_results.db'))
rundb['ewok_1004'] = ResultsDb(database=os.path.join(rootDir, 'ewok_1004_scheduler_results.db'))

help(rundb['enigma_1189'].getMetricId)
help(rundb['enigma_1189'].getMetricDisplayInfo)
help(rundb['enigma_1189'].getSummaryStats)
help(rundb['enigma_1189'].getPlotFiles)

metricName = 'NVisits'
mIds = {}
for r in rundb:
    mIds[r] = rundb[r].getMetricId(metricName=metricName)
    print r, mIds[r]
    print ''

# Retrieve all summary statistics for a metric + set of metric metadata + for a particular slicer.
metricName = 'NVisits'
metricMetadata = 'i band, WFD'
mIds = {}
for r in rundb:
    mIds[r] = rundb[r].getMetricId(metricName=metricName, metricMetadata=metricMetadata)
    print r, mIds[r]

for r in rundb:
    plotFiles = rundb[r].getPlotFiles(mIds[r])
    summaryStats = rundb[r].getSummaryStats(mIds[r])
    print "Run %s" %r
    print plotFiles['plotFile']  # this is a numpy array with the metric information + plot file name
    print summaryStats
    print ''

metricName = 'NVisits'
slicerName = 'OneDSlicer'
metricMetadata  = 'Per night'  # capitalization matters!
summaryStatName = 'Median'

stats = {}
for r in rundb:
    mIds = rundb[r].getMetricId(metricName=metricName, metricMetadata=metricMetadata, slicerName=slicerName)
    stats[r] = rundb[r].getSummaryStats(mIds, summaryName=summaryStatName)   

# All of the values in stats
print stats['enigma_1189']
# And the relevant 'summaryValue' -- of which there is only one, because we used one metricID and one summaryStatName.
print stats['enigma_1189']['summaryValue']

# So you can easily create bigger tables or ratios:
baseline = stats['enigma_1189']['summaryValue'][0]
for r in rundb:
    print r, stats[r]['summaryValue'][0], stats[r]['summaryValue'][0]/baseline

# Or you could pull out several summary statistics, to plot together.

# Nice names for the comparisons we'll do (nice names for a plot)
metricComparisons = ['Nights in survey', 'Total NVisits', 'NVisits Per night', 'Mean slew time', 'Mean Nfilter changes', 
                     'Median Nvisits WFD', 'Median Nvisits r All']
# But we need to know how to pull this info out of the resultsDB, so get the actual metric names, metadata, summaryName.
metricInfo = [{'metricName':'Total nights in survey', 'metadata':None, 'summary':None},
              {'metricName':'TotalNVisits', 'metadata':'All Visits', 'summary':None},
              {'metricName':'NVisits', 'metadata':'Per night', 'summary':'Median'},
              {'metricName':'Mean slewTime', 'metadata':None, 'summary':None},
              {'metricName':'Filter Changes', 'metadata':'Per night', 'summary':'Mean'}, 
              {'metricName':'Nvisits, all filters', 'metadata':'All filters WFD: histogram only', 'summary':'Median'},
              {'metricName':'Nvisits', 'metadata':'r band, all props', 'summary':'Median'}]

stats = {}
for r in rundb:
    stats[r] = np.zeros(len(metricComparisons), float)
    for i, (mComparison, mInfo) in enumerate(zip(metricComparisons, metricInfo)):
        mIds = rundb[r].getMetricId(metricName=mInfo['metricName'], metricMetadata=mInfo['metadata'])
        s = rundb[r].getSummaryStats(mIds, summaryName=mInfo['summary'])
        stats[r][i] = s['summaryValue'][0]        
    print r, stats[r]

# Because the scales will be quite different (# of visits vs. # of filter changes, for example), normalize
#   both by dividing by the first set of values (or pick another baseline).

baseline = stats['ewok_1004']
xoffset = 0.8/(float(len(rundb)))
x = np.arange(len(baseline))
colors = np.random.random_sample((len(baseline), 3))
for i, r in enumerate(rundb):
    plt.bar(x+i*xoffset, stats[r]/baseline, width=xoffset, color=colors[i], label=r)
plt.xticks(x, metricComparisons, rotation=60)
plt.axhline(1.0)
plt.ylim(0.9, 1.1)
plt.legend(loc=(1.0, 0.2))

import pandas as pd

metrics = {}
stats = {}
for r in rundb:
    metrics[r] = rundb[r].getMetricDisplayInfo()
    stats[r] = rundb[r].getSummaryStats()
    metrics[r] = pd.DataFrame(metrics[r])
    stats[r] = pd.DataFrame(stats[r])

# Let's pull out all the metrics for subgroup 'WFD' in the Seeing and SkyBrightness groups.

groupList = ['G: Seeing', 'H: SkyBrightness']

compareStats = {}
for r in rundb:
    m = metrics[r].query('displaySubgroup == "WFD"')
    m = m.query('displayGroup in @groupList')
    m = m.query('slicerName != "OneDSlicer"')
    m = m[m.metricName.str.contains('%ile') == False]
    mIds = m.metricId
    compareStats[r] = stats[r].query('metricId in @mIds')  #we could have done this using getSummaryStats too.

# Find stats in common. 
baseline =  'ewok_1004'
foundStat = np.ones(len(compareStats[baseline]), dtype=bool)
plotStats = {}
for r in rundb:
    plotStats[r] = np.zeros(len(compareStats[baseline]), float)
for count, (i, compStat) in enumerate(compareStats[baseline].iterrows()):
    for r in rundb:
        query = '(metricName == @compStat.metricName) and (metricMetadata == @compStat.metricMetadata)'
        query +=  ' and (summaryName == @compStat.summaryName)'
        s = compareStats[r].query(query)
        if len(s) > 0:
            s = s.iloc[0]
            plotStats[r][count] = s.summaryValue
        else:
            foundStat[count] = False
for r in rundb:
    plotStats[r] = plotStats[r][np.where(foundStat)]
print len(plotStats[baseline])
    
compareStats[baseline].loc[:,'foundCol'] = foundStat

baseline = compareStats['ewok_1004'].query('foundCol == True')
metricNames = []
for i, pStat in baseline.iterrows():
    if pStat.summaryName == 'Identity':
        metricNames.append(' '.join([pStat.metricName, pStat.metricMetadata]))
    else:
        metricNames.append(' '.join([pStat.summaryName, pStat.metricName, pStat.metricMetadata]))


baseline= 'ewok_1004'
xoffset = 0.8/(float(len(rundb)))
x = np.arange(len(plotStats[baseline]))
colors = np.random.random_sample((len(plotStats[baseline]), 3))
plt.figure(figsize=(20, 6))
for i, r in enumerate(rundb):
    plt.bar(x+i*xoffset, plotStats[r]/plotStats[baseline], width=xoffset, color=colors[i], label=r)
plt.xticks(x, metricNames, rotation=60)
plt.axhline(1.0)
plt.ylim(0.9, 1.1)
plt.legend(loc=(1.0, 0.2))



