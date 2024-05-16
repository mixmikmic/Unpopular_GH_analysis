import pickle
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *

def row_examination(df):
    counts = df.groupby(['LAT', 'LONG', 'year', 'month', 'day']).count()
    print 'Max. number of rows per lat/long coordinate: ', counts.max()[0]
    print 'Min. number of rows per lat/long coordinate: ', counts.min()[0]
    print 'Mean number of rows per lat/long coordinate: ', counts.mean()[0]

def conf_levels_examination(df): 
    print 'Confidence level info: ', df['CONF'].describe()

for year in xrange(2015, 2002, -1): 
    with open('../../../data/pickled_data/MODIS/df_' + str(year) + '.pkl') as f: 
        df = pickle.load(f)
        print 'Year: ', str(year)
        print '-' * 50, '\n'
        row_examination(df)
        conf_levels_examination(df)

def examine_index(df, index): 
    print df.query('LAT == @index[0] & LONG == @index[1] & year == @index[2] & month == @index[3] & day == @index[4]')

def examine_lown_rows(df, count_num, output = False): 
    fires_counts = df.groupby(['LAT', 'LONG', 'year', 'month', 'day']).count()['AREA'] == count_num
    if output: 
        for index in fires_counts[fires_counts == True].index[0:10]:
            print '-' * 50
            examine_index(df, index)
    else:  
        return fires_counts.index[0:10]

for year in xrange(2015, 2014, -1): 
    with open('../../../data/pickled_data/MODIS/df_' + str(year) + '.pkl') as f: 
        df = pickle.load(f)
        examine_lown_rows(df, 2, True)     

for year in xrange(2015, 2014, -1): 
    with open('../../../data/pickled_data/MODIS/df_' + str(year) + '.pkl') as f: 
        df = pickle.load(f)
        for row_number in [2, 4, 10]: 
            print 'Row Number', str(row_number)
            print '-' * 50
            rows_less = examine_lown_rows(df, row_number, False)
            df['LAT'] = df['LAT'].astype(float)
            df['LONG'] = df['LONG'].astype(float)
            for dist_out in [0.001, 0.01, 0.05, 0.1]:
                print 'dist', str(dist_out)
                print '-' * 50
                for index in rows_less: 
                    lat_1, lat_2 = float(index[0]) - dist_out, float(index[0]) + dist_out
                    long_1, long_2 = float(index[1]) - dist_out, float(index[1]) + dist_out
                    result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2')
                    print result.shape[0], index[0], index[1]

fires = [('Long Draw', 42.392, -117.894), ('Holloway', 41.973, -118.366), ('Mustang Complex', 45.425, -114.59), 
        ('Rush', 40.621, -120.152), ('Ash Creek MT', 45.669, -106.469)]

with open('../../../data/pickled_data/MODIS/df_' + str(2012) + '.pkl') as f: 
    df = pickle.load(f)
    df['LAT'] = df['LAT'].astype(float)
    df['LONG'] = df['LONG'].astype(float)
    for fire in fires: 
        print fire[0]
        print '-' * 50
        lat_orig, long_orig = fire[1], fire[2]
        for dist_out in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            lat_1, lat_2 = lat_orig - dist_out, lat_orig + dist_out 
            long_1, long_2 = long_orig - dist_out, long_orig + dist_out
            result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2')
            print 'Dist_out: %s' %str(dist_out), result.shape[0], '\n'

with open('../../../data/pickled_data/MODIS/df_' + str(2012) + '.pkl') as f: 
    df = pickle.load(f)
    df['LAT'] = df['LAT'].astype(float)
    df['LONG'] = df['LONG'].astype(float)
    df = df.set_index(['LAT', 'LONG'])
    indices = df.index
    unique_indices = np.unique(indices)
    num_indices = len(unique_indices)
    obs_array = []
    rand_indices = np.random.randint(low=0, high=num_indices, size=100)
    for index in rand_indices: 
        lat_orig, long_orig = unique_indices[index]
        num_obs = []
        for dist_out in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            lat_1, lat_2 = lat_orig - dist_out, lat_orig + dist_out 
            long_1, long_2 = long_orig - dist_out, long_orig + dist_out
            result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2')
            num_obs.append(result.shape[0])
        obs_array.append(num_obs)

np.array(obs_array).mean(axis = 0)

# Let's just add a datetime instance to this for our purposes, and then we can reuse the code. If this weren't eda and 
# I wasn't in an ipython notebook, I would have written this all into a function. 
fires = [('Long Draw', 42.392, -117.894, '2012-07-09'), ('Holloway', 41.973, -118.366, '2012-08-08'), 
         ('Mustang Complex', 45.425, -114.59, '2012-08-17'), ('Rush', 40.621, -120.152, '2012-08-12'), 
         ('Ash Creek MT', 45.669, -106.469, '2012-06-27')]

with open('../../../data/pickled_data/MODIS/df_' + str(2012) + '.pkl') as f: 
    df = pickle.load(f)
    year, month, day = df['year'], df['month'], df['day']
    df['datetime'] = pd.Series([pd.to_datetime(str(year) + '-' + str(month) + '-' + str(day)) for year, month, day in zip(year, month, day)])
    df['datetime'] = pd.Series([datetime.date() for datetime in df['datetime']])
    df['LAT'] = df['LAT'].astype(float)
    df['LONG'] = df['LONG'].astype(float)
    for fire in fires: 
        print fire[0]
        print '-' * 50
        lat_orig, long_orig = fire[1], fire[2]
        date = fire[3]
        for dist_out in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            for date_out in xrange(0, 6): 
                date_beg, date_end = pd.to_datetime(date).date(), (pd.to_datetime(date) + DateOffset(weeks=date_out)).date()
                lat_1, lat_2 = lat_orig - dist_out, lat_orig + dist_out 
                long_1, long_2 = long_orig - dist_out, long_orig + dist_out
                result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2') 
                result2 = result[(result['datetime'] >= date_beg) & (result['datetime'] <= date_end)] 
                if date_out == 0: 
                    print 'Dist_out: %s' %str(dist_out), 'Weeks Out: %s' %str(date_out), result.shape[0], '\n', date_end
                else: 
                    print 'Dist_out: %s' %str(dist_out), 'Weeks Out: %s' %str(date_out), result2.shape[0], '\n', date_end
            print '-' * 50

with open('../../../data/pickled_data/MODIS/df_' + str(2012) + '.pkl') as f: 
    df = pickle.load(f)
    year, month, day = df['year'], df['month'], df['day']
    df['datetime'] = pd.Series([pd.to_datetime(str(year) + '-' + str(month) + '-' + str(day)) for year, month, day in zip(year, month, day)])
    df['datetime'] = pd.Series([datetime.date() for datetime in df['datetime']])
    df['LAT'] = df['LAT'].astype(float)
    df['LONG'] = df['LONG'].astype(float)
    for fire in fires: 
        print fire[0]
        print '-' * 50
        lat_orig, long_orig = fire[1], fire[2]
        date = fire[3]
        for dist_out in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]: 
            for date_out in xrange(0, 6): 
                date_beg, date_end = (pd.to_datetime(date)- DateOffset(weeks=1)).date(), (pd.to_datetime(date) + DateOffset(weeks=date_out)).date()
                lat_1, lat_2 = lat_orig - dist_out, lat_orig + dist_out 
                long_1, long_2 = long_orig - dist_out, long_orig + dist_out
                result = df.query('LAT > @lat_1 & LAT < @lat_2 & LONG > @long_1 & LONG < @long_2') 
                result2 = result[(result['datetime'] >= date_beg) & (result['datetime'] <= date_end)] 
                if date_out == 0: 
                    print 'Dist_out: %s' %str(dist_out), 'Weeks Out: %s' %str(date_out), result.shape[0], '\n', date_end
                else: 
                    print 'Dist_out: %s' %str(dist_out), 'Weeks Out: %s' %str(date_out), result2.shape[0], '\n', date_end
            print '-' * 50





