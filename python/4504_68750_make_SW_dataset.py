get_ipython().magic('matplotlib inline')

import sys, os
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io.matlab import loadmat
import h5py

# date and time stuff
from datetime import datetime, timedelta
from dateutil import parser, relativedelta

import xray
import seaborn as sns

import sys
sys.path.append('/Users/nicolasf/CODE/paleopy/')

from paleopy import markov

types = ['SW1','SW2', 'SW3', 'SW4', 'SW5', 'SW6']

dict_types = dict(zip(types,range(1,len(types)+14 )))
inv_dict_types = {v: k for k, v in dict_types.items()}

inv_dict_types

# %%writefile 
def select_season(data, season='DJF', complete=True, start = 1948, end = 2014, rm_leap=False):
    from calendar import monthrange
    """
    Select a season from data
    data must be a Pandas Series or DataFrame with a datetime index
    """
        
    seasons_params = {}
    seasons_params['DJF'] = (12,2)
    seasons_params['JFM'] = (1,3)
    seasons_params['FMA'] = (2,4)
    seasons_params['MAM'] = (3,5)
    seasons_params['AMJ'] = (4,6)
    seasons_params['MJJ'] = (5,7)
    seasons_params['JJA'] = (6,8)
    seasons_params['JAS'] = (7,9)
    seasons_params['ASO'] = (8,10)
    seasons_params['SON'] = (9,11)
    seasons_params['OND'] = (10,12)
    seasons_params['NDJ'] = (11,1)
    seasons_params['Warm Season (Dec. - May)'] = (12, 5)
    seasons_params['Cold Season (Jun. - Nov.)'] = (6, 11)
    seasons_params['Year (Jan. - Dec.)'] = (1, 12)
    seasons_params['Hydro. year (Jul. - Jun.)'] = (7, 6)    
        
    ### defines the selector 
    selector = ((data.index.month >= seasons_params[season][0]) | (data.index.month <= seasons_params[season][1]))
    
    ### selects
    data = data[selector]
    
    ### if complete == True, we only select COMPLETE seasons 
    data = data.truncate(before='%s-%s-1' % (start, seasons_params[season][0]),                   after='%s-%s-%s' % (end, seasons_params[season][1], monthrange(end,seasons_params[season][1])[1] ))
    
    if rm_leap: 
        data[(data.index.month == 2) & (data.index.day == 29)] = np.nan
        data.dropna(inplace=True)
    
    return data

fname = '/Users/nicolasf/research/NIWA/paleo/Agent_based/data/kmeans_6_class_clusters_TS_Ldomain.csv'

sw_types = pd.read_csv(fname, parse_dates=True, index_col=0, names=['class'], header=None, skiprows=1)

sw_types.head()

# sw_types = sw_types.ix['1981':'2010']

types

dict_types = dict(zip(types,range(0,len(types)+1)))
inv_dict_types = {v: k for k, v in dict_types.items()}

maptypes = lambda x: inv_dict_types[x]

sw_types['type'] =  sw_types.applymap(maptypes)

sw_types.tail()

sw_types.to_csv('../data/SWPac_Types.csv')

lseason = ['AMJ',
 'ASO',
 'DJF',
 'FMA',
 'JAS',
 'JFM',
 'JJA',
 'MAM',
 'MJJ',
 'NDJ',
 'OND',
 'SON',
 'Cold Season (Jun. - Nov.)',
 'Warm Season (Dec. - May)',
 'Hydro. year (Jul. - Jun.)',
 'Year (Jan. - Dec.)']

if os.path.exists('../data/simulations_SWPac_types.hdf5'): 
    os.remove('../data/simulations_SWPac_types.hdf5')

f = h5py.File('../data/simulations_SWPac_types.hdf5', mode='a')

for season in lseason: 
    # calculates the probabilities over the climatological period (1981 - 2010)
    kseason = select_season(sw_types, start=1981, end=2010, season=season, rm_leap=False)
    probs = markov.get_probs(kseason['type'].values, types)
    probs = pd.Series(probs, index=types)
    classes, transition_matrix = markov.get_transition_probs(kseason['type'])
    probs = probs.reindex(classes)
    dict_classes, sim2D = markov.simulate_2D(classes, probs.values, transition_matrix, N=len(kseason), P=1000)
    probs = np.empty((len(classes), sim2D.shape[1]))
    for i in range(sim2D.shape[1]): 
        p = markov.get_probs(sim2D[:,i], np.arange(len(classes)))
        probs[:,i] = p
    f["/{}/probs".format(season)] = probs
    f["/{}/probs".format(season)].attrs['shape'] = '(class, simulation)'
    f["/{}/probs".format(season)].attrs['classes'] = ','.join(list(dict_classes.values()))
    del(probs, p)

f.close()

f.keys()

f = h5py.File('../outputs/simulations_SWPac_types.hdf5', mode='r')

probs = f['DJF']['probs'].value * 100

probs.shape

lseason

inv_dict_types

sc = ['-',' ','(',')','.']
for season in lseason: 
    print(season)
    season_title = season
    for c in sc: 
        season_title = season_title.replace(c,'_')
    probs = f[season]['probs'].value * 100
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(17,11))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = axes.flatten()
    for i in range(6): 
        ax = axes[i]
        p = probs[i,:]
        mp = p.mean()
        pu = np.percentile(p, 97.5)
        pl = np.percentile(p, 2.5)

        sns.distplot(p, ax=ax, color='#1B216B', kde_kws={'color':'coral'})
        ax.set_title(inv_dict_types[i+1],fontdict={'weight':'bold'})
        ax.axvline(mp, color='#1B216B')
        ax.axvline(pu, color='#13680D')
        ax.axvline(pl, color='#13680D')
        ax.text(0.01, 0.9, "mean = {:3.1f}\nu. perc. = {:3.1f}\nl. perc. = {:3.1f}".format(mp,pu,pl),               transform=ax.transAxes, bbox=dict(facecolor ='w'))
        [l.set_rotation(90) for l in ax.xaxis.get_ticklabels()]
    fig.savefig('/Users/nicolasf/Desktop/distr_SWPac_simus_{}.png'.format(season_title), dpi=200)
    plt.close(fig)
    

f.close()



