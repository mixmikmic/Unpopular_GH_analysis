import pickle
import os
import pandas as pd

os.chdir('/Users/sallamander/galvanize/forest-fires/data/pickled_data/MODIS/')

with open('df_2001.pkl') as f: 
    df_2001 = pickle.load(f)

df_2001.sort(['year', 'month', 'day', 'LAT', 'LONG'], inplace=True)

lat_long_df = pd.DataFrame(df_2001.groupby(['LAT', 'LONG']).count()['AREA']).reset_index().rename(columns={'AREA': 'COUNT'})

# Check shape before merging to make sure it remains the same. 
df_2001.shape

lat_long_2001_df = df_2001.merge(lat_long_df, on=['LAT', 'LONG'])

lat_long_2001_df.shape

lat_long_2001_df.head(5)

lat_long_2001_df.query('COUNT >=4')

lat_long_df['LAT'] = lat_long_df['LAT'].astype(float)
lat_long_df['LONG'] = lat_long_df['LONG'].astype(float)

type(lat_long_df['LAT'][0])

lat_long_2001_df.query('LAT > 42.18 & LAT < 43.18 & LONG < -111.094 & LONG > -112.094').sort(['year', 'month', 'day'])

lat_long_2001_df.query('LAT > 42.679 & LAT < 42.681 & LONG < -111.593 & LONG > -111.595').sort(['year', 'month', 'day'])

os.chdir('/Users/sallamander/galvanize/forest-fires/data/pickled_data/MODIS/')

with open('df_2015.pkl') as f: 
    df_2015 = pickle.load(f)

lat_long_df = pd.DataFrame(df_2015.groupby(['LAT', 'LONG']).count()['AREA']).reset_index().rename(columns={'AREA': 'COUNT'})

lat_long_2015_df = df_2015.merge(lat_long_df, on=['LAT', 'LONG'])

lat_long_2015_df['COUNT'].max()

lat_long_2015_df.query('COUNT >=10').sort(['LAT', 'LONG', 'year', 'month', 'day'])

lat_long_2015_df.query('COUNT >=3 & COUNT <= 5').sort(['LAT', 'LONG', 'year', 'month', 'day'])

lat_long_2015_df['AREA'].describe()



