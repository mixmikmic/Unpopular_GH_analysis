from dsfuncs.processing import remove_outliers
from dsfuncs.dist_plotting import plot_var_dist
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# I'll only be looking at certain columns, so let's only read in the ones that we'll actually be looking at. 
keep_cols = ['fire_bool']
for days_back in (0, 1, 2, 3, 4, 5, 6, 7, 365, 730, 1095): 
    keep_cols.append('all_nearby_count' + str(days_back))
    keep_cols.append('all_nearby_fires' + str(days_back))

engineered_df = pd.read_csv('../../modeling/model_input/geo_time_done.csv', usecols=keep_cols)
engineered_df.columns

keep_cols.remove('fire_bool') # We don't want this in there when we cycle through each of the columns to plot.
non_fires = engineered_df.query('fire_bool == 0')
fires = engineered_df.query('fire_bool == 1')

for col in keep_cols: 
    print 'Variable: {} : Non-fires, then fires'.format(col)
    print '-' * 50
    f, axes = plt.subplots(1, 8, figsize=(20, 5))
    plot_var_dist(non_fires[col], categorical=False, ax=axes[0:4], show=False)
    plot_var_dist(fires[col], categorical=False, ax=axes[4:], show=False)
    plt.show()

for col in keep_cols: 
    print 'Variable: {} : Non-fires, then fires'.format(col)
    print '-' * 50
    print 
    print non_fires[col].describe()
    print 
    print fires[col].describe()
    print '\n' * 3



