# builtin modules
import json
import logging
import datetime 
import io
import pathlib

# numeric 
import numpy as np
import pandas as pd
import netCDF4

# downloading data
import requests

# timezones
from dateutil.relativedelta import relativedelta
import pytz

# progress
from tqdm import tqdm_notebook as tqdm

# plotting
import matplotlib.dates 
import matplotlib.pyplot as plt

# tide
import utide


# for interactive charts
from ipywidgets import interact

get_ipython().magic('matplotlib inline')

# create a logger
logger = logging.getLogger('notebook')

# note that there are two stations for IJmuiden.
# The station was moved from the sluices to outside of the harbor in 1981.
ids = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD', 'IJMDBTHVN', 'VLISSGN']
names = {
    'DELFZL': 'Delfzijl',
    'DENHDR': 'Den Helder',
    'HARLGN': 'Harlingen',
    'HOEKVHLD': 'Hoek van Holland',
    'IJMDBTHVN': 'IJmuiden',
    'VLISSGN': 'Vlissingen'
}
# ids from http://www.psmsl.org/data/obtaining/
psmsl_ids = {
    'DELFZL': 24, 
    'DENHDR': 23, 
    'HARLGN': 25, 
    'HOEKVHLD': 22,
    'IJMDBTHVN': 32, 
    'VLISSGN': 20
    
}
current_year = 2017
# fill in the format later
path = str(
    pathlib.Path('~/src/sealevel/data/waterbase/{station}-{year}.txt').expanduser()
)


storm_surge_reports = [
    {
        'date': datetime.datetime(2017, 1, 14),
        'url': 'https://waterberichtgeving.rws.nl/water-en-weer/verwachtingen-water/water-en-weerverwachtingen-waternoordzee/stormvloedrapportages/stormvloedverslagen/download:782'
    },
    {
        'date': datetime.datetime(2017, 10, 29),
        'url': 'https://waterberichtgeving.rws.nl/water-en-weer/verwachtingen-water/water-en-weerverwachtingen-waternoordzee/stormvloedrapportages/stormvloedverslagen/download:994'
    }
]

# create a list of records
records = []
# for each station
for station in ids:
    # look back a few years for consistency (from-through)
    for year in range(current_year-2, current_year + 1):
        df = pd.read_csv(path.format(station=station, year=year), skiprows=3, sep=';')
        # there should be no missings
        assert df['waarde'].isna().sum() == 0
        # all values should be within this range
        # if not check what's happening
        assert df['waarde'].min() > -400
        assert df['waarde'].max() < 600
        # and check the units
        assert (df['eenheid'] == 'cm').all()

        mean = df['waarde'].mean()
        records.append({
            'station': station,
            'year': year,
            'mean': mean
        })
        

# merge all the records to get a list of mean sea level per year
latest_df = pd.DataFrame(records)
# check the mean for 2017
latest_df.set_index(['station', 'year']) 

# read the latest data 
sources = {}
for station in ids:
    df = pd.read_csv(path.format(station=station, year=current_year), skiprows=3, sep=';')
    df['date'] = pd.to_datetime(df['datum'] + ' ' + df['tijd'])
    # Several stations contain duplicates, drop them and keep the first
    # Not sure why.... (contact RWS)
    df = df.drop_duplicates(keep='first')
    # there should be no missings
    assert df['waarde'].isna().sum() == 0
    # all values should be within this range
    # if not check what's happening
    assert df['waarde'].min() > -400
    assert df['waarde'].max() < 600
    # and check the units
    assert (df['eenheid'] == 'cm').all()
    sources[station] = df


# this is the data, a bit verbose but the relevant things are datum tijd and waarde
df = sources[ids[0]].set_index('date')
# make sure the dataset is complete until (almost) end of the year
df.tail()

tides = {}
coefs = {}
for station, df in sources.items():
    # use date as an index (so we have a datetime index)
    df = df.set_index('date')
    t = matplotlib.dates.date2num(df.index.to_pydatetime())
    coef = utide.solve(
        t, 
        df['waarde'].values, # numpy array
        lat=52, # for the equilibrium nodal tide
        method='ols', # just use linear model
        conf_int='linear'
    )
    coefs[station] = coef
    tide = utide.reconstruct(t, coef)
    tides[station] = tide
    

for station, df in sources.items():
    tide = tides[station]
    # update dataframe (inline)   
    df['tide'] = tide['h']
    df['surge'] = df['waarde'] - df['tide']

# compute the maximum water levels
records = []
for station, df in sources.items():
    date_df = df.set_index('date')
    
    max_date = date_df['waarde'].idxmax()
    record = {
        'station': station, 
        'date': max_date, 
        'value': date_df['waarde'].loc[max_date]
    }
    records.append(record)
annual_maxima_df = pd.DataFrame(records)
annual_maxima_df

# compute the maximum surge
records = []
for station, df in sources.items():
    df = df.drop_duplicates(['date'])
    date_df = df.set_index('date')
    
    max_date = date_df['surge'].idxmax()
    record = {
        'station': station, 
        'date': max_date, 
        'surge': date_df['surge'].loc[max_date]
    }
    records.append(record)
annual_maxima_surge_df = pd.DataFrame(records)
annual_maxima_surge_df


fig, axes = plt.subplots(
    # 2 rows, 1 column
    3, 1, 
    # big
    figsize=(18, 16), 
    # focus on tide
    gridspec_kw=dict(height_ratios=[3, 1, 1]),
    sharex=True
)
for station, df in sources.items():
    index = df.set_index('date').index
    axes[0].plot(index.to_pydatetime(), df['waarde'], '-', label=names[station], linewidth=0.2)
    axes[1].plot(index.to_pydatetime(), df['tide'], '-', label=station, alpha=0.5, linewidth=0.3)
    axes[2].plot(index.to_pydatetime(), df['surge'], '-', label=station, alpha=0.5, linewidth=0.3)
axes[0].legend(loc='best');
axes[0].set_ylabel('water level [cm]')
axes[1].set_ylabel('astronomical tide [cm]')
axes[2].set_ylabel('surge [cm]')
for event in storm_surge_reports:
    axes[2].fill_between(
        [event['date'] + datetime.timedelta(hours=-48), event['date'] + datetime.timedelta(hours=48)],
        y1=axes[1].get_ylim()[0],
        y2=axes[1].get_ylim()[1],
        alpha=0.1,
        facecolor='black'
    )

# plot a window of a week

def plot(weeks=(0, 51)):
    
    fig, axes = plt.subplots(
        # 2 rows, 1 column
        3, 1, 
        # big
        figsize=(12, 8), 
        # focus on tide
        gridspec_kw=dict(height_ratios=[3, 1, 1]),
        sharex=True
    )
    for station, df in sources.items():
        selected = df[
            np.logical_and(
                df['date'] >= datetime.datetime(2017, 1, 1) + datetime.timedelta(weeks=weeks),
                df['date'] < datetime.datetime(2017, 1, 1) + datetime.timedelta(weeks=weeks + 1)
            )
        ]
        index = selected.set_index('date').index
        axes[0].plot(index.to_pydatetime(), selected['waarde'], '-', label=names[station], alpha=0.5, linewidth=2)
        axes[1].plot(index.to_pydatetime(), selected['tide'], '-', label=station, alpha=0.5, linewidth=2)
        axes[2].plot(index.to_pydatetime(), selected['surge'], '-', label=station, alpha=0.5, linewidth=2)
    axes[0].legend(loc='best');
    axes[0].set_ylabel('water level [cm]')
    axes[1].set_ylabel('astronomical tide [cm]')
    axes[2].set_ylabel('surge [cm]')
    axes[0].set_ylim(-300, 500)
    axes[1].set_ylim(-250, 250)
    axes[2].set_ylim(-100, 300)

interact(plot);

# now get the PSMSL data for comparison
psmsls = {}

# TODO: read the zip file
for station, id_ in psmsl_ids.items():
    df = pd.read_csv(io.StringIO(requests.get('http://www.psmsl.org/data/obtaining/met.monthly.data/{}.metdata'.format(
        id_
    )).text), sep=';', names=[
        'year', 'level', 'code', 'quality'
    ])
    df['year'] = df.year.apply(lambda x: np.floor(x).astype('int'))
    df['station'] = station
    psmsls[station] = df
psmsl_df = pd.concat(psmsls.values())
# compute sea level in cm
psmsl_df['sea_level'] = psmsl_df['level'] / 10

# compare data to metric data
# some differences exist
# see HKV report from 2017 on this topic 
# most differences are due to that I think hourly measurements are used for the psmsl mean
for station, df in psmsls.items():
    print(station)
    annual_df = df[['year', 'level']].groupby('year').mean()
    print(annual_df.tail(n=5))
    new_records = latest_df[np.logical_and(
        latest_df.station == station, 
        np.isin(latest_df.year, (2015, 2016, 2017))
    )]
    print(new_records)
    

# mean sealevel from psmsl
mean_df = psmsl_df[['year', 'sea_level']].groupby('year').mean()

mean_df.loc[current_year] = latest_df[latest_df['year'] == current_year]['mean'].mean()

# show the top 10 of highest sea levels
mean_df.sort_values('sea_level', ascending=False).head(n=10)

# Use the fitted values from the sea-level monitor (note that these are RLR not in NAP)
years = mean_df.index[mean_df.index > 1890]
# use the model without wind (otherwise the intercept does not match up)
fitted = (
    1.9164 * (years - 1970)  + 
    -25.7566  +
    7.7983 * np.cos(2*np.pi*(years-1970)/18.613) +
    -10.5326 * np.sin(2*np.pi*(years-1970)/18.613)  
)

fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(mean_df.index, mean_df['sea_level'])
ax.plot(years, fitted/10)
ax.set_ylabel('sea-surface height (w.r.t. NAP/RLR) in cm')
ax.set_xlabel('time [year]');

# find the maximum sea-level
mean_df.idxmax()

# check the current phase of nodal tide, u,v from sea-level monitor (full model)
tau = np.pi * 2
t = np.linspace(current_year - 18, current_year + 18, num=100)
nodal_tide = 7.5367*np.cos(tau*(t - 1970)/18.6) + -10.3536*np.sin(tau*(t - 1970)/18.6) 
amplitude = np.sqrt(7.5367**2 + (-10.3536)**2)

fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(t, nodal_tide/10);
ax.set_ylabel('nodal tide [cm]')
ax.fill_between([2017, 2018], *ax.get_ylim(), alpha=0.2)
ax.grid('on')

# next peak of nodal tide
2004.5 + 18.6

