import requests
import pandas as pd
import config   ## File with API key

api_key = config.bea_key

# Components of request
base = 'https://www.bea.gov/api/data/?&UserID={}'.format(api_key)
get_param = '&method=GetParameterValues'
dataset = '&DataSetName=GDPbyIndustry'
param = 'TableID'

# Construct URL from parameters above
url = '{}{}{}&ParameterName={}&ResultFormat=json'.format(base, get_param, dataset, param)

# Request parameter information from BEA API
r = requests.get(url).json()

# Show the results as a table:
pd.DataFrame(r['BEAAPI']['Results']['ParamValue']).set_index('Key')

param = 'Industry'

# Construct URL from parameters above
url = '{}{}{}&ParameterName={}&ResultFormat=json'.format(base, get_param, dataset, param)

# Request parameter information from BEA API
r = requests.get(url).json()

# Show the results as a table:
pd.DataFrame(r['BEAAPI']['Results']['ParamValue']).set_index('Key')

m = '&method=GetData'
ind = '&TableId=25'
freq = '&Frequency=A'
year = '&Year=ALL'
fmt = '&ResultFormat=json'
indus = '&Industry=23'  # Construction Industry

# Combined url for request
url = '{}{}{}{}{}{}{}{}'.format(base, m, dataset, year, indus, ind, freq, fmt)

r = requests.get(url).json()

df = pd.DataFrame(r['BEAAPI']['Results']['Data'])
df = df.replace('Construction', 'Gross Output')
df = df.set_index([pd.to_datetime(df['Year']), 'IndustrYDescription'])['DataValue'].unstack(1)
df = df.apply(pd.to_numeric)
df.tail()

df['Emp_sh'] = df['Compensation of employees'] / df['Gross Output']
df['Surplus_sh'] = df['Gross operating surplus'] / df['Gross Output']

get_ipython().run_line_magic('matplotlib', 'inline')
df[['Emp_sh', 'Surplus_sh']].plot(title='Employee & profit share of gross output')

