import requests
import pandas as pd

import config
key = config.census_key

base = 'http://api.census.gov/data/'
years = ['2015']#['2009', '2012', '2015']
variables = {'NAME':'Name',
             'B01001_001E': 'Population total',
             'B19013_001E': 'Real Median Income',}
v = ','.join(variables.keys())
c = '*'
s = '*'

df = pd.DataFrame()
for y in years:
    url = '{}{}/acs5?get={}&for=county:{}&in=state:{}&key={}'.format(
        base, y, v, c, s, key)
    r = requests.get(url).json()
    dft = pd.DataFrame(r[1:], columns=r[0])
    dft['Year'] = y
    df = df.append(dft)
df = df.rename(columns=variables).set_index(
    ['Name', 'Year']).sort_index(level='Name')
df.head()

df['Real Median Income'] = df['Real Median Income'].astype(float)

df['FIPS'] = df['state'] + df['county']
df['FIPS'] = df['FIPS'].astype(int)
df['FIPS'] = df['FIPS'].map(lambda i: str(i).zfill(5))
# County FIP Codes that have changed:
df['FIPS'] = df['FIPS'].str.replace('46102', '46113')

# For mapping results
import vincent
vincent.core.initialize_notebook()

geo_data = [{'name': 'counties',
             'url': 'geo/us_counties.topo.json',
             'feature': 'us_counties.geo'},            
            {'name': 'states',
             'url': 'geo/us_states.topo.json',
             'feature': 'us_states.geo'}
             ]

vis = vincent.Map(data=df, geo_data=geo_data, scale=1100,
                  projection='albersUsa', data_bind='Real Median Income',
                  data_key='FIPS', map_key={'counties': 'properties.FIPS'})

del vis.marks[1].properties.update
vis.marks[0].properties.enter.stroke.value = '#fff'
vis.marks[1].properties.enter.stroke.value = '#000000'
vis.scales['color'].domain = [0, 75000] # Adjust
vis.legend(title='Real Median Income')
vis.to_json('geo/vega.json')

vis.display()



