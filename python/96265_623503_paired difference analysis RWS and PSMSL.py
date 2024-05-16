import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# these come with python
import io
import zipfile
import functools
import datetime

# for downloading
import nbformat
import requests
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext pycodestyle_magic')

def execute_notebook(nbfile):
    with io.open(nbfile, encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)

    ip = get_ipython()

    for cell in tqdm(nb.cells):
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.source)

hdf_file = Path('stationData.h5')
if not hdf_file.is_file():
    print('create stationData.h5 first, uncomment line below and re-run cell (execution notebook takes approx. 2 hours)')
    # execute_notebook('retrieve timeseries from webservice RWS.ipynb')
else:
    print('file stationData.h5 is already created, please continue')

if hdf_file.is_file():
    hdf = pd.HDFStore(str(hdf_file))  # depends on PyTables
    keys = hdf.keys()
    print('established connection to stationData.h5')
else:
    print('file stationData.h5 is not created, did you run previous code block')

execute_notebook('retrieve timeseries from webservice PSMSL.ipynb')
# resulted DataFrame available as: df_psmsl

columns = ['VLISSGN']
index_year = pd.date_range('1890', '2018', freq='A')
index_month = pd.date_range('1890', '2018', freq='M')
df_year = pd.DataFrame(index=index_year, columns=columns, dtype=float)
df_month = pd.DataFrame(index=index_month, columns=columns, dtype=float)

def convert_raw_df(hdf, columns, index_year, index_month, df_year, df_month):
    df_raw = pd.DataFrame()
    for station in columns:
        for year in tqdm(index_year):
            key = '/'+station+'/year'+str(year.year)
            if key in keys:
                if hdf[key].isnull().sum()[0] == hdf[key].size:
                    print('key {0} contains only 0 values')
                df_raw = df_raw.append(hdf[key])
                annual_mean = hdf[key].astype(float).mean()[0]
                monthly_mean = hdf[key].resample('M').mean()
                df_year.set_value(str(year), station, annual_mean)
                df_month.loc[monthly_mean.index,
                             station] = monthly_mean.as_matrix().flatten()
    return(df_raw, df_year, df_month)

# data is stored hdfstore as separate table for each year for each station
# merge all years for single station
columns = ['VLISSGN']
df_raw, df_year, df_month = convert_raw_df(hdf, columns, index_year,
                                           index_month, df_year, df_month)

# histogram for the raw values to get insight in the distribution 
# over the period requested
xmin = datetime.datetime(1890,1,1)
xmax = datetime.datetime(2009,12,31)

df_raw.columns = ['Histogram of raw data at station Vlissingen']
ax = df_raw[(df_raw.index>xmin) & (df_raw.index<xmax)].hist()
ax.flatten()[0].set_xlabel("cm+NAP")
plt.show()

# since working with df_month introduces spikes at the annual change,
# we resample to monthly mean values using the raw dataframe.
df_raw.columns = columns
df_raw_M = df_raw.resample('M').mean()

df_raw_M.head()

def get_sel_seriesRWS(df_raw_M, xmin, xmax):
    raw_dates = []
    for raw_date in df_raw_M.loc[xmin:xmax].index:
        rw_date = pd.Timestamp(raw_date.year, raw_date.month, 15)
        raw_dates.append(rw_date)
    series_raw_dates = pd.Series(raw_dates)

    new_series_RWS = pd.Series(data=df_raw_M.loc[xmin:xmax].values.flatten(),
                               index=series_raw_dates.values)
    return new_series_RWS

# range of period without major spikes
xmin = datetime.datetime(1890, 4, 1)
xmax = datetime.datetime(2017, 4, 1)
series_raw_M = get_sel_seriesRWS(df_raw_M, xmin, xmax)
plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(df_psmsl.index, df_psmsl['VLISSINGEN_WATHTE_cmNAP'])
plt.ylabel('cm+NAP')
plt.ylim(-50, 30)
plt.xlim(xmin, xmax)
plt.title('PSMSL monthly metric values (cm+NAP for Vlissingen)')

plt.subplot(312)
plt.plot(series_raw_M.index, df_raw_M['VLISSGN'].loc[xmin:xmax])
plt.ylim(-50, 30)
plt.xlim(xmin, xmax)
plt.ylabel('cm+NAP')
plt.title('Webservice RWS monthly values (cm+NAP for Vlissingen)')

plt.subplot(313)
dif_series = (series_raw_M -
              df_psmsl['VLISSINGEN_WATHTE_cmNAP'].loc[xmin:xmax])
plt.plot(dif_series.index, dif_series.values)
plt.ylim(-25, 25)
plt.xlim(xmin, xmax)
plt.ylabel('cm')
plt.title('Difference PSMSL and RWS for Vlissingen (cm)')

plt.tight_layout()

xmin = datetime.datetime(1952, 1, 1)
xmax = datetime.datetime(1953, 12, 31)
series_raw_M = get_sel_seriesRWS(df_raw_M, xmin, xmax)
plt.figure(figsize=(12, 8))

plt.subplot(311)
plt.plot(df_psmsl.index, df_psmsl['VLISSINGEN_WATHTE_cmNAP'])
plt.axvspan(datetime.datetime(1953, 1, 1), datetime.datetime(1953, 4, 1),
            facecolor='#2ca02c', alpha=0.2)
plt.ylabel('cm+NAP')
plt.ylim(-50, 30)
plt.xlim(xmin, xmax)
plt.title('PSMSL monthly metric values (cm+NAP for Vlissingen)')

plt.subplot(312)
plt.plot(series_raw_M.index, series_raw_M.values)
plt.axvspan(datetime.datetime(1953, 1, 1), datetime.datetime(1953, 4, 1),
            facecolor='#2ca02c', alpha=0.2)
plt.ylim(-50, 30)
plt.xlim(xmin, xmax)
plt.ylabel('cm+NAP')
plt.title('Webservice RWS monthly values (cm+NAP for Vlissingen)')

plt.subplot(313)
dif_series = (series_raw_M -
              df_psmsl['VLISSINGEN_WATHTE_cmNAP'].loc[xmin:xmax])
plt.plot(dif_series.index, dif_series.values)
plt.axvspan(datetime.datetime(1953, 1, 1), datetime.datetime(1953, 4, 1),
            facecolor='#2ca02c', alpha=0.2)
plt.ylim(-25, 25)
plt.xlim(xmin, xmax)
plt.ylabel('cm')
plt.title('Difference PSMSL and RWS for Vlissingen (cm)')
plt.tight_layout()

# first create a helper function to extraxt the mean, max and
# series containing the absolute difference given two stations 
# and a period
def getdif_maxmean(col_RWS, col_PSMSL, xmin, xmax,
                   hdf, columns, index_year, index_month,
                   df_year, df_month):
    columns = [col_RWS]
    df_raw, df_year, df_month = convert_raw_df(hdf, columns,
                                               index_year, index_month,
                                               df_year, df_month)
    df_raw_M = df_raw.resample('M').mean()

    series_raw_M = get_sel_seriesRWS(df_raw_M, xmin, xmax)
    dif_series = (series_raw_M - df_psmsl[col_PSMSL].loc[xmin:xmax])
    return abs(dif_series).mean(), abs(dif_series).max(), dif_series

df_diff_sel_stations = pd.DataFrame()

xmin = datetime.datetime(1890, 1, 1)
xmax = datetime.datetime(2017, 12, 31)

abs_dif_mean_list = []
abs_dif_max_list = []
name_list = []

cols_RWS = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD',
            'IJMDBTHVN', 'VLISSGN']
cols_PSMSL = ['DELFZIJL_WATHTE_cmNAP', 'DEN HELDER_WATHTE_cmNAP',
              'HARLINGEN_WATHTE_cmNAP', 'HOEK VAN HOLLAND_WATHTE_cmNAP',
              'IJMUIDEN_WATHTE_cmNAP', 'VLISSINGEN_WATHTE_cmNAP']

for idx, col_RWS in enumerate(tqdm(cols_RWS)):
    col_PSMSL = cols_PSMSL[idx]
    print('station PSMSL {0} - station RWS {1}'.format(col_PSMSL, col_RWS))
    # get the data from both datasets
    abs_dif_mean, abs_dif_max, dif_series = getdif_maxmean(col_RWS, col_PSMSL,
                                                           xmin, xmax, hdf,
                                                           columns, index_year,
                                                           index_month, df_year,
                                                           df_month)
    # append to new lists and overview dataframe
    abs_dif_mean_list.append(abs_dif_mean)
    abs_dif_max_list.append(abs_dif_max)
    name_list.append(col_RWS)
    dif_series.name = col_RWS
    df_diff_sel_stations = pd.concat((df_diff_sel_stations, dif_series),
                                     axis=1)

station_difs = pd.DataFrame([abs_dif_mean_list, abs_dif_max_list],
                            columns=name_list,
                            index=['absolute mean difference (cm)',
                                   'absolute max difference (cm)'])

axes = df_diff_sel_stations.plot(subplots=True, legend=True,
                                 figsize=(12, 10), sharex=True,
                                 title='Absolute difference PSMSL and RWS')
for ax in axes:
    ax.set(ylabel='cm')
plt.tight_layout()

print(station_difs.mean(axis=1))
station_difs.head()

