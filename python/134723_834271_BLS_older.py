get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import requests
import json

api_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

# API key in config.py which contains: bls_key = 'key'
import config
key = '?registrationkey={}'.format(config.bls_key)

# Series stored as a dictionary
series_dict = {
    'LNS14000003': 'White', 
    'LNS14000006': 'Black', 
    'LNS14000009': 'Hispanic'}

# Start year and end year
date_r = (2000, 2017)

# Handle dates
dates = [(str(date_r[0]), str(date_r[1]))]
while int(dates[-1][1]) - int(dates[-1][0]) > 10:
    dates = [(str(date_r[0]), str(date_r[0]+9))]
    d1 = int(dates[-1][0])
    while int(dates[-1][1]) < date_r[1]:
        d1 = d1 + 10
        d2 = min([date_r[1], d1+9])
        dates.append((str(d1),(d2))) 

df = pd.DataFrame()

for start, end in dates:
    # Submit the list of series as data
    data = json.dumps({
        "seriesid": series_dict.keys(),
        "startyear": start, 
        "endyear": end})

    # Post request for the data
    p = requests.post(
        '{}{}'.format(api_url, key), 
        headers={'Content-type': 'application/json'}, 
        data=data).json()
    dft = pd.DataFrame()
    for s in p['Results']['series']:
        dft[series_dict[s['seriesID']]] = pd.Series(
            index = pd.to_datetime(
                ['{} {}'.format(
                    i['period'], 
                    i['year']) for i in s['data']]),
            data = [i['value'] for i in s['data']],
            ).astype(float).iloc[::-1]
    df = df.append(dft)        
# Output results
print 'Post Request Status: ' + p['status']
print df.tail(13)
df.plot(title='Unemployment Rates by Race or Origin')

