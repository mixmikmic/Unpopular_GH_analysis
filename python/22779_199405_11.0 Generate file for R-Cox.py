import os
import numpy as np
import pandas as pd
import xarray as xr

from pointprocess import *
from lightning_setup import *
get_ipython().magic('matplotlib inline')

c = Region(city=cities['cedar'])
c.SUBSETTED = False
c.CENTER = (37.7, -111.8)
c.RADIUS = 0.6
c.define_grid()

version1 = pd.HDFStore('./output/Version1/store.h5')

def dateparser(y, j, t):
    x = ' '.join([y, str(int(float(j))), t])
    return pd.datetime.strptime(x, '%Y %j %H:%M:%S')

df = pd.read_csv('./input/pwv147215720409387.txt', delim_whitespace=True, skiprows=[1], na_values=[-9.99],
                 parse_dates={'time': [1,2,3]}, date_parser=dateparser, index_col='time')

get_ipython().run_cell_magic('time', '', "\ndef get_FC(y):\n    ds = c.get_ds(y=y, filter_CG=dict(method='less_than', amax=-10), func=None)\n    df = ds.to_dataframe()\n    ds.close()\n    df.index = df.time\n    FC = df['lat'].resample('24H', base=12, label='right').count()\n    FC.name = 'FC'\n    return FC\n\nFC = get_FC(2010)\nfor y in range(2011,2016):\n    FC = pd.concat([FC, get_FC(y)])\n\nversion1['EPK_FC_2010_2015'] = FC")

IPW = df.resample('24H', base=12, label='right').mean()['IPW']

EPK_FC = version1['EPK_FC_2010_2015']
EPK_ngrids = c.gridx.shape[0]*c.gridy.shape[0]

# convert counts to density
df = pd.concat([IPW, EPK_FC], axis=1)
df.columns = ['IPW', 'FD']

# the data really should end in Sept 2015 when we run out of lightning data
df = df[:EPK_FC.index[-1]]

# NAN values should really be zeros
df['FD'] = df['FD'].fillna(0)

df.tail()

# convert to 0,1 with a threshold equal to 10 events per year
thresh = df.FD.sort_values(ascending=False)[50]

df['FD'][df['FD'] > thresh] = 1
df['FD'][df['FD'] <= thresh] = 0

from collections import OrderedDict

d = {'YYYY': df.index.year,
     'MM': df.index.month,
     'DD': df.index.day,
     'START': df.index.dayofyear-1,  # start at 0
     'STOP': df.index.dayofyear,
     'EVENT': df['FD'],  
     'X1': df['IPW']}

OD = OrderedDict([(k, d[k]) for k in ['YYYY', 'MM', 'DD', 'START', 'STOP', 'EVENT', 'X1']])
cox_df = pd.DataFrame(OD)

# if we want to do something about missing values, we can try interpolating. 
# cox_df = cox_df.interpolate()

cox_df.to_csv('cox_test.csv', index=False)

cox_df[cox_df['EVENT']>0].plot.scatter(y='X1', x='START', alpha=.2)

(cox_df['EVENT'] ==1).sum()

cox_df[(cox_df['START']>130) & (cox_df['START']<270)].plot.scatter(y='X1', x='START', alpha=.2)

