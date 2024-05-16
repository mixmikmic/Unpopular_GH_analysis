import pandas as pd
import os

os.chdir('/home/domestic-ra/Working/CPS_ORG/EPOPs/')

cols = ['year', 'female', 'age', 'educ', 'empl', 'orgwgt']
year = '2015'

df = pd.read_stata('Data/cepr_org_{}.dta'.format(year), columns=cols)

df = df[(df['female'] == 1) & (df['age'].isin(range(25,55)))]

epop = (df['orgwgt'] * df['empl']).sum() / df['orgwgt'].sum() * 100
print('CEPR ORG CPS: {}: Women, age 25-54: {:0.2f}'.format(year, epop))

import requests
import json
import config # file called config.py with my API key

series = 'LNU02300062'  # BLS Series ID of interest

# BLS API v1 url for series
url = 'https://api.bls.gov/publicAPI/v1/timeseries/data/{}'.format(series)
print(url)

# Get the data returned by the url and series id
r = requests.get(url).json()
print('Status: ' + r['status'])

# Generate pandas dataframe from the data returned
df2 = pd.DataFrame(r['Results']['series'][0]['data'])

epop2 = df2[df2['year'] == year]['value'].astype(float).mean()
print('BLS Benchmark: {}: Women, age 25-54: {:0.2f}'.format(year, epop2))

df2.set_index('year').loc['2017'].sort_values('periodName')

# Identify which columns to keep from the full CPS
cols = ['year', 'female', 'age', 'educ', 'empl', 'orgwgt', 'weekpay', 'uhourse']
df = pd.read_stata('Data/cepr_org_2009.dta', columns=cols)

dft = df[(df['female']==1) & (df['age'] > 15) & (df['uhourse'] > 34) & 
        (df['orgwgt'] > -1) & (df['empl'].notnull()) & (df['weekpay'] > 0)]

import wquantiles

wquantiles.median(dft['weekpay'], dft['orgwgt'])

round(dft['orgwgt'].sum() / 12000, 0)



