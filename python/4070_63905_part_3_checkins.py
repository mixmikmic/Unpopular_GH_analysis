import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

from helper_methods import read_json
df_chk = read_json('data/yelp_academic_dataset_checkin.json')
df_chk.head(3)

df_chk.info()

df_chk.business_id.unique().size

def sum_checkins(day_index, dct):
    return sum([value for key, value in dct.iteritems() if '-' + str(day_index) in key])

dayofweek = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
for i, day in enumerate(dayofweek):
    df_chk[day + '_checkins'] = df_chk.checkin_info.apply(lambda d: sum_checkins(i, d))
df_chk['total_checkins'] = df_chk.checkin_info.apply(lambda d: sum(d.values()))
df_chk['num_bins'] = df_chk.checkin_info.apply(len)

df_chk.head(3)

df_chk.describe()

plt.hist(df_chk.total_checkins, bins=25, range=(0, 2500), log=True)
plt.xlabel('Total checkins')
plt.ylabel('Count')

df = pd.read_csv('data/training_labels.txt')
df['weighted_violations'] = 1 * df['*'] + 3 * df['**'] + 5 * df['***']
df.head()

avg_violations = df.groupby('restaurant_id').agg({'*': [np.size, np.mean, np.sum], '**': [np.mean, np.sum], '***': [np.mean, np.sum], 'weighted_violations': [np.mean, np.sum]})
avg_violations.head(3)

avg_violations.info()

from helper_methods import biz2yelp
trans = biz2yelp()
trans.columns = ['restaurant_id', 'business_id']
trans.head()

trans_df = pd.merge(trans, avg_violations, left_on='restaurant_id', right_index=True, how='inner')
violations_checkins = pd.merge(trans_df, df_chk, on='business_id', how='inner')
violations_checkins.head(3)

violations_checkins.info()

plt.plot(violations_checkins.total_checkins, violations_checkins[('weighted_violations', 'mean')], '.')
plt.xlim(0, 2000)
plt.xlabel('Total checkins')
plt.ylabel('Weighted violations')

from scipy.stats import pearsonr
pearsonr(violations_checkins.total_checkins, violations_checkins[('weighted_violations', 'mean')])

