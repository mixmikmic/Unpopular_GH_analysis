get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

get_ipython().system('unzip -u -d ../../data ../../data/crime_incident_data_2013.zip')
get_ipython().system('mv ../../data/crime_incident_data.csv ../../data/crime_incident_data_2013.csv')
get_ipython().system('unzip -u -d ../../data ../../data/crime_incident_data_2014.zip')
get_ipython().system('mv ../../data/crime_incident_data.csv ../../data/crime_incident_data_2014.csv')

import pandas as pd
crime = {}
for yr in range(13, 15):
    crime[yr] = pd.DataFrame.from_csv('crime_incident_data_20{}.csv'.format(yr))
df = crime[13]
df

df.describe()

get_ipython().system('pip install pandas-profiling')
import pandas_profiling as prof
stats = prof.describe(df)
stats

df = crime[13].copy()
labels = list(df.columns)
for i in range(-1, -3, -1):
    labels[i] = labels[i].strip().lower()[0]
df = pd.DataFrame(df.values, columns=labels)
plt.scatter(df.x, df.y)

