import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

df = pd.read_csv('FL_insurance_sample.csv')
df.head(3).transpose()

df.info()

#select avg(point_latitude), min(point_latitude), max(point_latitude), stddev(point_latitude) from insur;
df.describe()

# need periscope or something to make plots in SQL
plt.plot(df.point_longitude, df.point_latitude, '.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Florida insurance samples')

# select county, count(county) from insur group by county order by count(county) desc;
df.county.value_counts()

df.construction.value_counts()

# the average total insured value increase from 2011 to 2012 by construction material
df['tiv_diff'] = df.tiv_2012 - df.tiv_2011

#select construction, avg(tiv_2012 - tiv_2011) from insur group by construction;
df.groupby('construction').agg({'tiv_diff':np.mean})

# select policy_id, tiv_2011, tiv_2012, construction from insur where construction='wood' and (tiv_2012 - tiv_2011 > 250000.0);
df[['policyID', 'tiv_2011', 'tiv_2012', 'construction']][(df.construction=='Wood') & (df.tiv_diff > 250000.0)].sort_values('policyID')

df[df.county.str.contains('[A-Z]', regex=True)][['county']].head()

df[['policyID', 'tiv_2011', 'tiv_2012', 'construction']][df.construction.isin(['Steel Frame', 'Reinforced Concrete'])].head()

df[['policyID', 'tiv_2011', 'tiv_2012', 'construction']].query('tiv_2011 > tiv_2012 and construction == \'Steel Frame\'').head()

