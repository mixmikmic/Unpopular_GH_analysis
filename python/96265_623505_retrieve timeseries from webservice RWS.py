import requests
import json

import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

from multiprocessing.pool import ThreadPool
from tqdm import tqdm_notebook as tqdm
from time import time as timer
from IPython.display import clear_output

get_ipython().magic('matplotlib inline')

collect_catalogus = ('https://waterwebservices.rijkswaterstaat.nl/' +
                     'METADATASERVICES_DBO/' +
                     'OphalenCatalogus/')
collect_observations = ('https://waterwebservices.rijkswaterstaat.nl/' +
                        'ONLINEWAARNEMINGENSERVICES_DBO/' +
                        'OphalenWaarnemingen')
collect_latest_observations = ('https://waterwebservices.rijkswaterstaat.nl/' +
                               'ONLINEWAARNEMINGENSERVICES_DBO/' +
                               'OphalenLaatsteWaarnemingen')

# get station information from DDL (metadata uit Catalogus)
request = {
    "CatalogusFilter": {
        "Eenheden": True,
        "Grootheden": True,
        "Hoedanigheden": True
    }
}
resp = requests.post(collect_catalogus, json=request)
result = resp.json()
# print all variables in the catalogus
# print(result)

df_locations = pd.DataFrame(result['LocatieLijst']).set_index('Code')
# load normalized JSON object (since it contains nested JSON)
df_metadata = pd.io.json.json_normalize(
    result['AquoMetadataLijst']).set_index('AquoMetadata_MessageID')

# note that there are two stations for IJmuiden.
# The station was moved from the sluices to outside of the harbor in 1981.
ids = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD',
       'IJMDBTHVN', 'IJMDNDSS', 'VLISSGN']
df_locations.loc[ids]

request_manual = {
  'Locatie': {
    'X': 761899.770959577,
    'Y': 5915790.48491405,
    'Code': 'DELFZL'
  },
  'AquoPlusWaarnemingMetadata': {
    'AquoMetadata': {
      'Eenheid': {
        'Code': 'cm'
      },
      'Grootheid': {
        'Code': 'WATHTE'
      },
      'Hoedanigheid': {
        'Code': 'NAP'
      }
    }
  },
  'Periode': {
    'Einddatumtijd': '2012-01-27T09:30:00.000+01:00',
    'Begindatumtijd': '2012-01-27T09:00:00.000+01:00'
  }
}

aqpwm = request_manual['AquoPlusWaarnemingMetadata']
unit = aqpwm['AquoMetadata']['Eenheid']['Code']
quantity = aqpwm['AquoMetadata']['Grootheid']['Code']
qualitiy = aqpwm['AquoMetadata']['Hoedanigheid']['Code']
column = unit+'_'+quantity+qualitiy

try:
    resp = requests.post(collect_observations, json=request_manual)
    df_out = pd.io.json.json_normalize(
        resp.json()['WaarnemingenLijst'][0]['MetingenLijst'])
    df_out = df_out[['Meetwaarde.Waarde_Numeriek', 'Tijdstip']]
    df_out['Tijdstip'] = pd.to_datetime(df_out['Tijdstip'])
    df_out.set_index('Tijdstip', inplace=True)
    df_out.index.name = 'time'
    df_out.columns = [column]
    df_out.loc[df_out[column] == 999999999.0] = np.nan
    df_out.plot()
except Exception as e:
    print(e)

def strftime(date):
    """
    required datetime format is not ISO standard date format.
    current conversion method works, but improvements are welcome
    asked on SO, but no responses: https://stackoverflow.com/q/45610753/2459096
    """
    (dt, micro, tz) = date.strftime(
        '%Y-%m-%dT%H:%M:%S.%f%Z:00').replace('+', '.').split('.')
    dt = "%s.%03d+%s" % (dt, int(micro) / 1000, tz)
    return dt


def POST_collect_measurements(start_datetime, df_location, df_aquo_metadata):
    """
    create a JSOB object for a POST request for collection of observations

    Parameters
    ---
    start_datetime : datetime object inc tzinfo
        (end_datetime is hardcoded 1 month after start_datetime)
    df_location : dataframe
        with sinlge station location info
    df_aquo_metadata : dataframe
        with single unit/quantity/quality information

    Return
    ------
    JSON object
    """
    # empty json object
    request_dynamic = {}

    request_dynamic['Locatie'] = {}
    rd_location = request_dynamic['Locatie']
    rd_location['X'] = df_location.X
    rd_location['Y'] = df_location.Y
    rd_location['Code'] = df_location.name

    request_dynamic['AquoPlusWaarnemingMetadata'] = {}
    rd_apwm = request_dynamic['AquoPlusWaarnemingMetadata']
    rd_apwm['AquoMetadata'] = {}
    rd_aquo_metadata = rd_apwm['AquoMetadata']
    rd_aquo_metadata['Eenheid'] = {
        'Code': df_aquo_metadata['Eenheid.Code'].values[0]}
    rd_aquo_metadata['Grootheid'] = {
        'Code': df_aquo_metadata['Grootheid.Code'].values[0]}
    rd_aquo_metadata['Hoedanigheid'] = {
        'Code': df_aquo_metadata['Hoedanigheid.Code'].values[0]}

    request_dynamic['Periode'] = {}
    rd_period = request_dynamic['Periode']
    rd_period['Begindatumtijd'] = strftime(start_datetime)
    rd_period['Einddatumtijd'] = strftime(start_datetime +
                                          relativedelta(months=1))

    return request_dynamic

# create a long list of data objects
# only use start-dates since end-date is always 1 month after the start-date
start_dates = []
for year in np.arange(1890, 2018):
    for month in np.arange(1, 13):
        start_dates.append(datetime(year=year,
                                    month=month,
                                    day=1,
                                    hour=0,
                                    minute=0,
                                    tzinfo=pytz.timezone('Etc/GMT-1')))
start_dates = pd.Series(start_dates)
# startDates.head()

sel_dates = start_dates[(start_dates > '1890-01-01') &
                        (start_dates < '1890-06-01')]
sel_dates

# select a single station
for station in ids[0:1]:
    df_location = df_locations.loc[station]
df_location.head()

# select a metadata object using the unit/quanity/quality
df_WATHTE_NAP = df_metadata[(df_metadata['Grootheid.Code'] == 'WATHTE') &
                            (df_metadata['Hoedanigheid.Code'] == 'NAP')]
df_WATHTE_NAP.T.head()

request_dynamic = POST_collect_measurements(start_datetime=sel_dates[3],
                                            df_location=df_location,
                                            df_aquo_metadata=df_WATHTE_NAP)
request_dynamic

def fetch_collect_obersvations(start_date, column_name):
    try:
        # prepare the POST object
        request_dynamic = POST_collect_measurements(
            start_datetime=start_date,
            df_location=df_location,
            df_aquo_metadata=df_WATHTE_NAP)
        # do the query
        resp = requests.post(collect_observations, json=request_dynamic)

        # parse the result to DataFrame
        df_out = pd.io.json.json_normalize(
            resp.json()['WaarnemingenLijst'][0]['MetingenLijst'])
        df_out = df_out[['Meetwaarde.Waarde_Numeriek', 'Tijdstip']]
        df_out['Tijdstip'] = pd.to_datetime(df_out['Tijdstip'])
        df_out.set_index('Tijdstip', inplace=True)
        df_out.columns = [column_name]
        df_out.index.name = 'time'
        df_out.loc[df_out[column_name] == 999999999.0] = np.nan
        # add to HDFStore
        hdf.append(key=df_location.name + '/year'+str(start_date.year),
                   value=df_out, format='table')

        return start_date, None
    except Exception as e:
        return start_date, e

hdf = pd.HDFStore('stationData.h5')  # depends on PyTables
start = timer()

# itereer over stations
for station in tqdm(ids):
    df_location = df_locations.loc[station]

    for start_date in tqdm(start_dates):
        start_date, error = fetch_collect_obersvations(
            start_date,
            column_name=column)

        if error is None:
            print("%r fetched and processed in %ss" % (
                start_date, timer() - start))
        else:
            print("error fetching %r: %s" % (start_date, error))
        clear_output(wait=True)
print("Elapsed time: %s" % (timer() - start,))

hdf.close()

