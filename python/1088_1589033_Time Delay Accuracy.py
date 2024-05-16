import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Import MAF modules.
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
from lsst.sims.maf.metricBundles import MetricBundle, MetricBundleGroup
# Import the contributed metrics and stackers 
import mafContrib

runName = 'minion_1016'
database = runName + '_sqlite.db'
opsdb = db.OpsimDatabase(database)
outDir = 'tmp'

metric = mafContrib.TdcMetric(metricName='TDC', seasonCol='season', expMJDCol='expMJD', nightCol='night')
slicer = slicers.HealpixSlicer(nside=64, lonCol='ditheredRA', latCol='ditheredDec')

plotFuncs = [plots.HealpixSkyMap, plots.HealpixHistogram]
slicer.plotFuncs = plotFuncs

sqlconstraint = 'night < %i and filter = "i"' % (3*365.25)
tdcBundle = MetricBundle(metric=metric, slicer=slicer, sqlconstraint=sqlconstraint, runName=runName)

resultsDb = db.ResultsDb(outDir=outDir)
bdict = {'tdc':tdcBundle}
bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)

bgroup.runAll()

bdict.keys()

minVal = 0.01
maxVal = {'Accuracy':0.04, 'Precision':10.0, 'Rate':40, 'Cadence':14, 'Season':8.0, 'Campaign':11.0}
units = {'Accuracy':'%', 'Precision':'%', 'Rate':'%', 'Cadence':'days', 'Season':'months', 'Campaign':'years'}
for key in maxVal:
    plotDict = {'xMin':minVal, 'xMax':maxVal[key], 'colorMin':minVal, 'colorMax':maxVal[key]}
    plotDict['xlabel'] = 'TDC %s (%s)' % (key, units[key])
    bdict['TDC_%s' % (key)].setPlotDict(plotDict)

bgroup.plotAll(closefigs=False)

tdcBundle.metricValues

import numpy as np
x = tdcBundle.metricValues
index = np.where(x.mask == False)

f = np.array([each['rate'] for each in x[index]])
A = np.array([each['accuracy'] for each in x[index]])
P = np.array([each['precision'] for each in x[index]])
c = np.array([each['cadence'] for each in x[index]])
s = np.array([each['season'] for each in x[index]])
y = np.array([each['campaign'] for each in x[index]])
print np.mean(f), np.mean(A), np.mean(P), np.mean(c), np.mean(s), np.mean(y)

accuracy_threshold = 0.04 # 5 times better than threshold of 0.2% set by Hojjati & Linder (2014).

high_accuracy = np.where(A < accuracy_threshold)
high_fraction = 100*(1.0*len(A[high_accuracy]))/(1.0*len(A))
print "Fraction of total survey area providing high accuracy time delays = ",np.round(high_fraction,1),'%'

high_accuracy_cadence = np.median(c[high_accuracy])
print "Median night-to-night cadence in high accuracy regions = ",np.round(high_accuracy_cadence,1),'days'

high_accuracy_season = np.median(s[high_accuracy])
print "Median season length in high accuracy regions = ",np.round(high_accuracy_season,1),'months'

high_accuracy_campaign = np.median(y[high_accuracy])
print "Median campaign length in high accuracy regions = ",int(high_accuracy_campaign),'years'

Nside = 64
Npix = 12*Nside**2
Area_per_pixel = 4*np.pi / float(Npix) # steradians
Area_per_pixel *= (180.0/np.pi)*(180.0/np.pi) # square degrees
high_accuracy_area = len(A[high_accuracy])*Area_per_pixel
print "Area of sky providing high accuracy time delays = ",int(high_accuracy_area),"sq deg"

precision_per_lens = np.array([np.mean(P[high_accuracy]),4.0])
precision_per_lens = np.sqrt(np.sum(precision_per_lens*precision_per_lens))
print "Mean precision per lens in high accuracy sample, including modeling error = ",np.round(precision_per_lens,2),'%'

fraction = np.mean(f[high_accuracy])
N_lenses = int((high_accuracy_area/18000.0) * (fraction/30.0) * 400)
print "Number of lenses in high accuracy sample = ",N_lenses

distance_precision = (precision_per_lens * (N_lenses > 0)) / (np.sqrt(N_lenses) + (N_lenses == 0))
print "Maximum combined percentage distance precision (as in Coe & Moustakas 2009) = ",np.round(distance_precision,2),'%'

def evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', filters='ugrizy', Nyears=10):
    '''
    Sets up and executes a MAF analysis based on the Time Delay Challenge metrics.
    
    Parameters
    ----------
    runName : string ('minion_2016')
        The name of an OpSim simulation, whose output database will be used.
    filters : string ('ugrizy')
        List of bands to be used in analysis.
    Nyears : int
        No. of years in campaign to be used in analysis, starting from night 0.
        
    Returns
    -------
    results : dict
        Various summary statistics
    
    Notes
    -----

    '''
    # Set up some of the metadata, and connect to the OpSim database. 
    database = runName + '_sqlite.db'
    opsdb = db.OpsimDatabase(database)
    
    # Instantiate the metrics, stackers and slicer that we want to use. 
    # These are the TDC metrics, the season stacker, and the healpix slicer. 
    # Actually, since we'll just use the stackers in their default configuration, we don't need to 
    # explicitly instantiate the stackers -- MAF will handle that for us.  
    # Note that the metric (TdcMetric) is actually a "complex" metric, as it calculates A, P, and f 
    # all in one go (thus re-using the cadence/season/campaign values which must also be calculated
    # for each set of visits), and then has 'reduce' methods that separate each of these individual
    # results into separate values. 
    metric = mafContrib.TdcMetric(metricName='TDC', seasonCol='season', expMJDCol='expMJD', nightCol='night')
    slicer = slicers.HealpixSlicer(nside=64, lonCol='ditheredRA', latCol='ditheredDec')
    
    # Set the plotFuncs so that we only create the skymap and histogram for each metric result 
    # (we're not interested in the power spectrum). 
    plotFuncs = [plots.HealpixSkyMap, plots.HealpixHistogram]
    slicer.plotFuncs = plotFuncs
    
    # Write the SQL constraint:
    sql = 'night < %i' % (365.25*Nyears)
    sqlstring = str(Nyears)+'years-'
    if filters == 'ugrizy':
        sql += ''
        sqlstring += 'ugrizy'
    elif filters == 'ri':
        sql += ' and (filter="r" or filter="i")'
        sqlstring += 'r+i-only'
    else:
        raise ValueError('Unrecognised filter set '+filters)

    # Set the output directory name:
    outDir = 'output_'+runName+'_'+sqlstring
    
    # Now bundle everything up:
    tdcBundle = MetricBundle(metric=metric, slicer=slicer, sqlconstraint=sql, runName=runName)
    resultsDb = db.ResultsDb(outDir=outDir)
    bdict = {'tdc':tdcBundle}
    bgroup = MetricBundleGroup(bdict, opsdb, outDir=outDir, resultsDb=resultsDb)

    # And run the metrics!
    bgroup.runAll()
    
    # Now to make the plots. 
    # Note that we now have more bundles in our bundle dictionary - these new bundles contain 
    # the results of the reduce functions - so, A/P/f separately:
    #     bdict.keys() => ['tdc', 'TDC_Rate', 'TDC_Precision', 'TDC_Accuracy', 'TDC_Cadence', 'TDC_Campaign', 'TDC_Season']
    # We want to set the plotDict for each of these separately, so that we can get each plot 
    # to look "just right", and then we'll make the plots.   
    minVal = 0.01
    maxVal = {'Accuracy':0.04, 'Precision':10.0, 'Rate':40, 'Cadence':14, 'Season':8.0, 'Campaign':11.0}
    units = {'Accuracy':'%', 'Precision':'%', 'Rate':'%', 'Cadence':'days', 'Season':'months', 'Campaign':'years'}
    for key in maxVal:
        plotDict = {'xMin':minVal, 'xMax':maxVal[key], 'colorMin':minVal, 'colorMax':maxVal[key]}
        plotDict['xlabel'] = 'TDC %s (%s)' % (key, units[key])
        bdict['TDC_%s' % (key)].setPlotDict(plotDict)
    
    bgroup.plotAll(closefigs=False)
    
    # Now pull out metric values so that we can compute some useful summaries: 
    import numpy as np
    x = tdcBundle.metricValues
    index = np.where(x.mask == False)
    f = np.array([each['rate'] for each in x[index]])
    A = np.array([each['accuracy'] for each in x[index]])
    P = np.array([each['precision'] for each in x[index]])
    c = np.array([each['cadence'] for each in x[index]])
    s = np.array([each['season'] for each in x[index]])
    y = np.array([each['campaign'] for each in x[index]])

    # Summaries:
    results = dict()
    results['runName'] = runName
    results['filters'] = filters
    results['Nyears'] = Nyears
    
    accuracy_threshold = 0.04 # 5 times better than threshold of 0.2% set by Hojjati & Linder (2014).
    high_accuracy = np.where(A < accuracy_threshold)
    results['high_accuracy_area_fraction'] = 100*(1.0*len(A[high_accuracy]))/(1.0*len(A))
    print "Fraction of total survey area providing high accuracy time delays = ",np.round(results['high_accuracy_area_fraction'],1),'%'

    results['high_accuracy_cadence'] = np.median(c[high_accuracy])
    print "Median night-to-night cadence in high accuracy regions = ",np.round(results['high_accuracy_cadence'],1),'days'

    results['high_accuracy_season'] = np.median(s[high_accuracy])
    print "Median season length in high accuracy regions = ",np.round(results['high_accuracy_season'],1),'months'

    results['high_accuracy_campaign'] = np.median(y[high_accuracy])
    print "Median campaign length in high accuracy regions = ",int(results['high_accuracy_campaign']),'years'

    Nside = 64
    Npix = 12*Nside**2
    Area_per_pixel = 4*np.pi / float(Npix) # steradians
    Area_per_pixel *= (180.0/np.pi)*(180.0/np.pi) # square degrees
    results['high_accuracy_area'] = len(A[high_accuracy])*Area_per_pixel
    print "Area of sky providing high accuracy time delays = ",int(results['high_accuracy_area']),"sq deg"

    precision_per_lens = np.array([np.mean(P[high_accuracy]),4.0])
    results['precision_per_lens'] = np.sqrt(np.sum(precision_per_lens*precision_per_lens))
    print "Mean precision per lens in high accuracy sample, including modeling error = ",np.round(results['precision_per_lens'],2),'%'

    fraction = np.mean(f[high_accuracy])
    results['N_lenses'] = int((results['high_accuracy_area']/18000.0) * (fraction/30.0) * 400)
    print "Number of lenses in high accuracy sample = ",results['N_lenses']
 
    results['distance_precision'] = results['precision_per_lens'] / np.sqrt(results['N_lenses'])
    print "Maximum combined percentage distance precision (as in Coe & Moustakas 2009) = ",np.round(results['distance_precision'],2),'%'

    return results

results = []

results.append(evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', Nyears=10, filters='ugrizy'))

results.append(evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', Nyears=5, filters='ugrizy'))

results.append(evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', Nyears=10, filters='ri'))

results.append(evaluate_opsim_run_for_time_delay_performance(runName='minion_1016', Nyears=5, filters='ri'))

results.append(evaluate_opsim_run_for_time_delay_performance(runName='kraken_1043', Nyears=10, filters='ugrizy'))

results.append(evaluate_opsim_run_for_time_delay_performance(runName='kraken_1043', Nyears=5, filters='ugrizy'))

results.append(evaluate_opsim_run_for_time_delay_performance(runName='kraken_1043', Nyears=10, filters='ri'))

results.append(evaluate_opsim_run_for_time_delay_performance(runName='kraken_1043', Nyears=5, filters='ri'))

def make_latex_table(results):
    """
    Writes a latex table, with one row per test, presenting all the TDC metrics.
    
    Parameters
    ----------
    results : list(dict)
        List of results dictionaries, one per experiment.
        
    Returns
    -------
    None.
    
    Notes
    -----
    The latex code is written to a simple .tex file for \input into a document.
    
    Each element of the results list is a dictionary, like this:
    {'high_accuracy_season': 6.9118266805479518, 'high_accuracy_area': 19004.12600851645, 
     'high_accuracy_area_fraction': 70.67103620474407, 'precision_per_lens': 5.0872446140504994, 
     'N_lenses': 468, 'Nyears': 10, 'runName': 'minion_1016', 'filters': 'ugrizy', 
     'high_accuracy_cadence': 4.5279775970305893, 'distance_precision': 0.23515796547162932, 
     'high_accuracy_campaign': 10.0}
     
    The interpretation of these numbers is as follows:
    
    Fraction of total survey area providing high accuracy time delays =  70.7 %
    Median night-to-night cadence in high accuracy regions =  4.5 days
    Median season length in high accuracy regions =  6.9 months
    Median campaign length in high accuracy regions =  10 years
    Area of sky providing high accuracy time delays =  19004 sq deg
    Mean precision per lens in high accuracy sample, including modeling error =  5.09 %
    Number of lenses in high accuracy sample =  468
    Maximum combined percentage distance precision (as in Coe & Moustakas 2009) =  0.24 %

    "High accuracy" means Accuracy metric > 0.04 
    
    Which element of results is which?
    
    for k in range(len(results)):
       print k, results[k]['runName'], results[k]['filters'], results[k]['Nyears']

    0 minion_1016 ugrizy 10
    1 minion_1016 ugrizy 5
    2 minion_1016 ri 10
    3 minion_1016 ri 5
    4 kraken_1043 ugrizy 10
    5 kraken_1043 ugrizy 5
    6 kraken_1043 ri 10
    7 kraken_1043 ri 5

    """
    
    # Open file object: 
    texfile = 'table_lenstimedelays.tex'
    f = open(texfile, 'w')
    
    # Start latex:
    tex = r'''
\begin{table*}
\begin{center}
\caption{Lens Time Delay Metric Analysis Results.}
\label{tab:lenstimedelays:results}
\footnotesize
\begin{tabularx}{\linewidth}{ccccccccc}
  \hline
  \OpSim run                       % runName -> db
   & Filters                       % filters
    & Years                        % Nyears
     & \texttt{cadence}            % high_accuracy_cadence
      & \texttt{season}            % high_accuracy_season
       & \texttt{Area}             % high_accuracy_area
        & \texttt{dtPrecision}     % precision_per_lens
         & \texttt{Nlenses}        % N_lenses
          & \texttt{DPrecision} \\ % distance_precision
  \hline\hline'''
    f.write(tex)

    # Now write the table rows:
    for k in range(8):
        x = results[k]
        if x['runName'] == 'minion_1016':
            x['db'] = '\opsimdbref{db:baseCadence}'
        elif x['runName'] == 'kraken_1043':
            x['db'] = '\opsimdbref{db:NoVisitPairs}'
        else:
            raise ValueError('Unrecognized runName: '+x['runName'])
        tex = r'''
  {db}
   & ${filters}$
    & ${Nyears:.0f}$
     & ${high_accuracy_cadence:.1f}$
      & ${high_accuracy_season:.1f}$
       & ${high_accuracy_area:.0f}$
        & ${precision_per_lens:.2f}$
         & ${N_lenses:.0f}$
          & ${distance_precision:.2f}$ \\'''.format(**x)
        f.write(tex)
        
    # Now finish up the table:    
    tex = r'''
   \hline

\multicolumn{9}{p{\linewidth}}{\scriptsize Notes: see the text for
the definitions of each metric.}
\end{tabularx}
\normalsize
\medskip\\
\end{center}
\end{table*}'''
    
    # Write last part to file and close up:
    f.write(tex)
    f.close()
    
    # Report
    print "LaTeX table written to "+texfile
    
    return

make_latex_table(results)

get_ipython().system(' cat table_lenstimedelays.tex')

