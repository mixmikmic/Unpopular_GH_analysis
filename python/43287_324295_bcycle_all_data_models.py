import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# todo ! Define the style in one place to keep graphs consistent

# plt.style.use('fivethirtyeight')
# # plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Helvetica'
# plt.rcParams['font.monospace'] = 'Consolas'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['figure.titlesize'] = 12

PLT_DPI = 150


def plot_ts(df, true, pred, title, ax):
    '''Generates one of the subplots to show time series'''
    plot_df = df.resample('1D').sum()
    ax = plot_df.plot(y=[pred, true], ax=ax) # , color='black', style=['--', '-'])
    ax.set_xlabel('', fontdict={'size' : 14})
    ax.set_ylabel('Rentals', fontdict={'size' : 14})
    ax.set_title(title + ' time series', fontdict={'size' : 16}) 
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.legend(['Predicted rentals', 'Actual rentals'], fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)   
    

def plot_scatter(true, pred, title, ax):
    '''Plots the results of a validation run on a scatter plot'''
    min_val = result_val_df.min().min() - 10.0
    max_val = result_val_df.max().max() + 20.0

    plt.scatter(x=true, y=pred)
    plt.axis('equal')
    plt.axis([min_val, max_val, min_val, max_val])
    plt.plot([min_val, max_val], [min_val, max_val], color='k', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Actual rentals', fontdict={'size' : 14})
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.set_ylabel('Predicted rentals', fontdict={'size' : 14})
    ax.set_title(title, fontdict={'size' : 16}) 

    filename = title.lower().replace(' ', '_')

def plot_all_results(df, true, pred, title):
    ''''''
    fig, ax = plt.subplots(1,2,figsize=(20,10), gridspec_kw={'width_ratios':[2,1]})
    plot_ts(df, true, pred, title, ax=ax[0])
    plot_scatter(df[true], df[pred], title, ax[1])
    filename=title.lower().replace(' ', '-').replace(',','')
    plt.savefig(filename, type='png', dpi=PLT_DPI, bbox_inches='tight')
    print('Saved file to {}'.format(filename))
    

from bcycle_lib.all_utils import load_bcycle_data

print('Loading stations and trips....', end='')
stations_df, trips_df = load_bcycle_data('../input', 'all_stations_clean.csv', 'all_trips_clean.csv', verbose=False)
print('done!')
print('Bike trips loaded from {} to {}'.format(trips_df.index[0], trips_df.index[-1]))

print('\nStations DF info:')
stations_df.info()
print('\nTrips DF info:')
trips_df.info()

# import requests
# import io

# def weather_url_from_dates(start_date, end_date):
#     '''Creates a URL string to fetch weather data between dates
#     INPUT: start_date - start date for weather
#            end_date - end date for weather
#     RETURNS: string of the URL 
#     '''
#     assert start_date.year == end_date.year, 'Weather requests have to use same year'
    
#     url = 'https://www.wunderground.com/history/airport/KATT/'
#     url += str(start_date.year) + '/' 
#     url += str(start_date.month) + '/'
#     url += str(start_date.day) + '/'
#     url += 'CustomHistory.html?dayend=' + str(end_date.day)
#     url += '&monthend=' + str(end_date.month)
#     url += '&yearend=' + str(end_date.year)
#     url += '&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1'
    
#     return url

# def weather_from_df_dates(df, verbose=False):
#     '''Returns a dictionary of weather dataframes, one per year
#     INPUT: Dataframe with date index
#     RETURNS : Dataframe of corresponding weather information
#     '''
#     yearly_weather = list()
#     unique_years = set(trips_df.index.year)
#     sorted_years = sorted(unique_years, key=int)
    
#     for year in sorted_years:
#         year_df = trips_df[str(year)]
#         start_date = year_df.index[0]
#         end_date = year_df.index[-1]
#         year_url = weather_url_from_dates(start_date, end_date)
#         if verbose:
#             print('Year {}: start date {}, end date {}'.format(year, start_date, end_date))
# #             print('URL: {}'.format(year_url))

#         if verbose: print('Fetching CSV data ... ', end='')
#         req = requests.get(year_url).content
#         req_df = pd.read_csv(io.StringIO(req.decode('utf-8')))
#         yearly_weather.append(req_df)
#         if verbose: print('done')
            
#     combined_df = pd.concat(yearly_weather)
#     return combined_df

# weather_df = weather_from_df_dates(trips_df, verbose=True)
# print('weather_df shape: {}'.format(weather_df.shape))

# from bcycle_lib.all_utils import clean_weather
# # Let's check the data for missing values, and forward-fill
# # print('Initial Weather missing value counts:')
# # print(weather_df.isnull().sum(axis=0))

# for col in weather_df.columns:
#     if 'Events' not in col:
#         weather_df[col] = weather_df[col].fillna(method='pad')

# print('\nAfter forward-filling NA values (apart from Events):')
# print(weather_df.isnull().sum(axis=0))

# from bcycle_lib.all_utils import clean_weather

# weather_df = clean_weather(weather_df)
# weather_df.to_csv('../input/all_weather.csv')

from bcycle_lib.all_utils import clean_weather

weather_df = pd.read_csv('../input/all_weather.csv')
weather_df = weather_df.set_index('date')
weather_df.head()

from bcycle_lib.all_utils import add_time_features

TRAIN_START = '2014-01-01'
TRAIN_END = '2015-12-31'
VAL_START = '2016-01-01'
VAL_END = '2016-12-31'

hourly_trips_df = trips_df.resample('1H').size().to_frame(name='count')
hourly_trips_df = add_time_features(hourly_trips_df)
train_df = hourly_trips_df[TRAIN_START:TRAIN_END].copy()
val_df = hourly_trips_df[VAL_START:VAL_END].copy()

n_train = train_df.shape[0]
n_val = val_df.shape[0]
n_total = n_train + n_val
n_train_pct = (n_train / n_total) * 100.0
n_val_pct = (n_val / n_total) * 100.0

print('\nTraining data first and last row:\n{}\n{}'.format(train_df.index[0], train_df.index[-1]))
print('\nValidation data first and last row:\n{}\n{}\n'.format(val_df.index[0], val_df.index[-1]))

print('Train data shape: {}, {:.2f}% of rows'.format(train_df.shape, n_train_pct))
print('Validation data shape: {}, {:.2f}% of rows'.format(val_df.shape, n_val_pct))

train_df.head()

from bcycle_lib.all_utils import plot_lines

plot_df = train_df.resample('1D').sum()['count']

plot_lines(plot_df, plt.subplots(1,1,figsize=(20,10)), 
                                 title='Training set rentals', 
                                 xlabel='', ylabel='Hourly rentals')

# Let's plot the validation set
plot_df = val_df.resample('1D').sum()['count']

plot_lines(plot_df, plt.subplots(1,1,figsize=(20,10)), 
                                 title='Validation set rentals', 
                                 xlabel='Date', ylabel='Hourly rentals')

from bcycle_lib.all_utils import plot_hist
SIZE=(20,6)

plot_hist(train_df['count'], bins=50, size=SIZE, title='Training hourly trips', xlabel='', ylabel='Count')
plot_hist(val_df['count'], bins=50, size=SIZE, title='Validation hourly trips', xlabel='', ylabel='Count')

# First create a daily rentals dataframe, split it into training and validation
from bcycle_lib.all_utils import add_time_features

train_df = add_time_features(train_df)
val_df = add_time_features(val_df)

print('Training data shape: {}'.format(train_df.shape))
print('Validation data shape: {}'.format(val_df.shape))

# Now we need to split into X and y
from bcycle_lib.all_utils import reg_x_y_split

X_train, y_train, _ = reg_x_y_split(train_df[['day-hour', 'count']], 
                                           target_col='count', 
                                           ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_df[['day-hour', 'count']], 
                                     target_col='count', 
                                     ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from bcycle_lib.all_utils import df_from_results, plot_results, plot_val

reg = Ridge()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

scores_df = pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, index=['linreg_time'])

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                 val_df.index, y_val, y_val_pred)

print('Hour-of-day baseline RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day baseline')

# Create a list of national holidays, with their observed dates days around them
holidays = {'hol_new_year' : ('2014-01-01', '2015-01-01', '2016-01-01'),
            'hol_mlk' : ('2014-01-18', '2014-01-19','2014-01-20',
                         '2015-01-17', '2015-01-18','2015-01-19',
                         '2016-01-16', '2016-01-17','2016-01-18'),
            'hol_presidents' : ('2014-02-15', '2014-02-16', '2014-02-17',
                                '2015-02-14', '2015-02-15', '2015-02-16',
                                '2016-02-13', '2016-02-14', '2016-02-15'),
            'hol_memorial' : ('2014-05-24', '2014-05-25', '2014-05-26',
                              '2015-05-23', '2015-05-24', '2015-05-25',
                              '2016-05-28', '2016-05-29', '2016-05-30'),
            'hol_independence' : ('2014-07-04', '2014-07-05', '2014-07-06',
                                  '2015-07-03', '2015-07-04', '2015-07-05',
                                  '2016-07-02', '2016-07-03', '2016-07-04'),
            'hol_labor' : ('2014-08-30', '2014-08-31', '2014-09-01',
                           '2015-09-05', '2015-09-06', '2015-09-07',
                           '2016-09-03', '2016-09-04', '2016-09-05'),
            'hol_columbus' : ('2014-10-11', '2014-10-12', '2014-10-13',
                              '2015-10-10', '2015-10-11', '2015-10-12',
                              '2016-10-08', '2016-10-09', '2016-10-10'),
            'hol_veterans' : ('2014-11-11', '2015-11-11', '2016-11-11'),
            'hol_thanksgiving' : ('2014-11-27', '2014-11-28', '2014-11-29', '2014-11-30',
                                  '2015-11-26', '2015-11-27', '2015-11-28', '2015-11-29',
                                  '2016-11-24', '2016-11-25', '2016-11-26', '2016-11-27'),
            'hol_christmas' : ('2014-12-25', '2014-12-26', '2014-12-27', '2014-12-28',
                               '2015-12-25', '2015-12-26', '2015-12-27', '2015-12-28',
                               '2016-12-24', '2016-12-25', '2016-12-26', '2016-12-27')
           }


def add_date_indicator(df, col, dates):
    '''Adds a new indicator column with given dates set to 1
    INPUT: df - Dataframe
           col - New column name
           dates - Tuple of dates to set indicator to 1
    RETURNS: Dataframe with new column
    '''
    df.loc[:,col] = 0
    for date in dates:
        if date in df.index:
            df.loc[date, col] = 1
            
    df[col] = df[col].astype(np.uint8)
    return df

for key, value in holidays.items():
    train_df = add_date_indicator(train_df, key, value)
    val_df = add_date_indicator(val_df, key, value)

# Create a list of national holidays, with their observed dates days around them
import itertools

def day_list(start_date, end_date):
    '''Creates list of dates between `start_date` and `end_date`'''
    date_range = pd.date_range(start_date, end_date)
    dates = [d.strftime('%Y-%m-%d') for d in date_range]
    return dates

sxsw2014 = day_list('2014-03-07', '2014-03-16')
sxsw2015 = day_list('2015-03-13', '2015-03-22')
sxsw2016 = day_list('2016-03-11', '2016-03-20')
sxsw = list(itertools.chain.from_iterable([sxsw2014, sxsw2015, sxsw2016]))

acl2014_wk1 = day_list('2014-10-03', '2014-10-05')
acl2014_wk2 = day_list('2014-10-10', '2014-10-12')
acl2015_wk1 = day_list('2015-10-02', '2015-10-04')
acl2015_wk2 = day_list('2015-10-09', '2015-10-11')
acl2016_wk1 = day_list('2016-09-30', '2016-10-02')
acl2016_wk2 = day_list('2016-10-07', '2016-10-09')
acl = list(itertools.chain.from_iterable([acl2014_wk1, acl2014_wk2, 
                                          acl2015_wk1, acl2015_wk2, 
                                          acl2016_wk1, acl2016_wk2]))


events = {'event_sxsw' : sxsw,
          'event_acl'  : acl
         }

for key, value in events.items():
    train_df = add_date_indicator(train_df, key, value)
    val_df = add_date_indicator(val_df, key, value)

# Check they were all added here 1
train_df.describe()
# train_df.head()

# train_df[train_df['event_sxsw'] == 1]

# Now we need to split into X and y
from bcycle_lib.all_utils import reg_x_y_split

X_train, y_train, _ = reg_x_y_split(train_df,
                                    target_col='count', 
                                    ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_df,
                                target_col='count', 
                                ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

from sklearn.linear_model import Ridge

reg = Ridge()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_events' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
                                              index=['linreg_time_events']))

print('Hour-of-day and events RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day with events')

def add_lag_time_features(df, col):
    """Adds time-lagged features to improve prediction
    INPUT: df - Dataframe with date index
           col - column in dataframe used to calculate lags
    RETURNS: Dataframe with extra lag features
    """
#     df[col + '_lag_1H'] = df[col].shift(1).fillna(method='backfill')
    df[col + '_lag_1D'] = df[col].shift(24 * 1).fillna(method='backfill')
    df[col + '_lag_2D'] = df[col].shift(24 * 2).fillna(method='backfill')
    df[col + '_lag_3D'] = df[col].shift(24 * 3).fillna(method='backfill')
    df[col + '_lag_4D'] = df[col].shift(24 * 4).fillna(method='backfill')
    df[col + '_lag_5D'] = df[col].shift(24 * 5).fillna(method='backfill')
    df[col + '_lag_6D'] = df[col].shift(24 * 6).fillna(method='backfill')
    df[col + '_lag_1W'] = df[col].shift(24 * 7).fillna(method='backfill')
    return df

def add_win_time_features(df, col):
    """Adds rolling window features to improve prediction
    INPUT: df - Dataframe with date index
           col - column in dataframe used to calculate lags
    RETURNS: Dataframe with extra window features
    """
    df[col + '_win_1D'] = df[col].rolling(window=24, win_type='blackman').mean().fillna(method='backfill')
    df[col + '_win_1W'] = df[col].rolling(window=24*7, win_type='blackman').mean().fillna(method='backfill')
    return df

def add_median_time_features(df, col):
    """Adds median bike rental values to correct for longer term changes
    """
    df[col + '_med_1D'] = df[col].shift(24).resample('1D').median()
    df[col + '_med_1D'] = df[col + '_med_1D'].fillna(method='ffill').fillna(0)
    df[col + '_med_1W'] = df[col].shift(24*7).resample('1W').median()
    df[col + '_med_1W'] = df[col + '_med_1W'].fillna(method='ffill').fillna(0)
    df[col + '_med_1M'] = df[col].shift(24*30).resample('1M').median()
    df[col + '_med_1M'] = df[col + '_med_1M'].fillna(method='ffill').fillna(0)

    return df


train_df = add_lag_time_features(train_df, 'count')
val_df   = add_lag_time_features(val_df  , 'count')

train_df = add_win_time_features(train_df, 'count')
val_df   = add_win_time_features(val_df  , 'count')

train_df = add_median_time_features(train_df, 'count')
val_df   = add_median_time_features(val_df  , 'count')

# # Lag features

# plot_df = train_df['2015-01-01':'2015-01-31']
# plot_lines(plot_df[['count', 'count_win_1D']], plt.subplots(1,1,figsize=(20,10)), title='', xlabel='', ylabel='')

# train_df.loc['2014-01-01':'2014-01-31', ('count', 'count_weekly_median')].plot.line(figsize=(20,10))

# Now we need to split into X and y
from bcycle_lib.all_utils import reg_x_y_split

X_train, y_train, _ = reg_x_y_split(train_df,
                                    target_col='count', 
                                    ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_df,
                                target_col='count', 
                                ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

scores_df

from sklearn.linear_model import Ridge

reg = Ridge()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_events_lags' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
                                              index=['linreg_time_events_lags']))

print('Hour-of-day with events and lags RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day with events and lags')

# train_weather_df['precipitation'].plot.hist(bins=40, figsize=(20,10))

# # Merge the training and validation datasets with the weather dataframe

def merge_daily_weather(df, weather_df):
    '''Merges the dataframes using the date in their indexes
    INPUT: df - Dataframe to be merged with date-based index
           weather_df - Dataframe to be merged with date-based index
    RETURNS: merged dataframe
    '''    

    # Extract the date only from df's index
    df = df.reset_index()
    df['date'] = df['datetime'].dt.date.astype('datetime64')
#     df = df.set_index('datetime')
    
    # Extract the date field to join on
    weather_df = weather_df.reset_index()
    weather_df['date'] = weather_df['date'].astype('datetime64')
    
    # Merge with the weather information using the date
    merged_df = pd.merge(df, weather_df, on='date', how='left')
    merged_df.index = df.index
    merged_df = merged_df.set_index('datetime', drop=True)
    merged_df = merged_df.drop('date', axis=1)
    assert df.shape[0] == merged_df.shape[0], "Error - row mismatch after merge"
    
    return merged_df

GOOD_COLS = ['max_temp', 'min_temp', 'max_gust', 'precipitation', 
        'cloud_pct', 'thunderstorm']


train_weather_df = merge_daily_weather(train_df, weather_df[GOOD_COLS])
val_weather_df = merge_daily_weather(val_df, weather_df[GOOD_COLS])

train_weather_df.head()

# Now we need to split into X and y
from bcycle_lib.all_utils import reg_x_y_split

X_train, y_train, _ = reg_x_y_split(train_weather_df,
                                    target_col='count', 
                                    ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_weather_df,
                                target_col='count', 
                                ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

from sklearn.linear_model import Ridge

reg = Ridge()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_events_lags_weather' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
                                              index=['linreg_time_events_lags_weather']))

print('Hour-of-day, events, lags, and weather RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day with events, lags, and weather')

plot_all_results(result_train_df, 'true', 'pred', 'Training Hour-of-day with events, lags, and weather')

trips_df.head()

from bcycle_lib.all_utils import plot_scores
plot_scores(scores_df, 'Model scores', 'val_rmse')
plt.savefig('scores.png', dpi=PLT_DPI, bbox_inches='tight')

scores_df.round(2)

from sklearn.preprocessing import StandardScaler

def model_eval(model, train_df, val_df, verbose=False):
    '''Evaluates model using training and validation sets'''
    X_train, y_train, _ = reg_x_y_split(train_df, target_col='count',  ohe_cols=['day-hour'], verbose=verbose)
    X_val, y_val, _ = reg_x_y_split(val_df, target_col='count',  ohe_cols=['day-hour'], verbose=verbose)

    if verbose:
        print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
        print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
    
    reg = model
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_val_pred = reg.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                             val_df.index, y_val, y_val_pred)

    out = {'train_rmse' : train_rmse, 
           'val_rmse' : val_rmse,
           'result_train' : result_train_df,
           'result_val' : result_val_df}
    
    print('RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

    return out

model_result = model_eval(Ridge(), train_weather_df, val_weather_df,
               verbose=False)

#     plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day with events, lags, and weather')

# Ridge regression
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

cv_results = dict()

for alpha in (0.001, 0.01, 0.1, 1, 10, 100):
    ridge = Ridge(alpha=alpha)
    model_result = model_eval(ridge, train_weather_df, val_weather_df, verbose=False)
    cv_results[alpha] = model_result


best_val_rmse = 100    
for key, value in cv_results.items():
    if (cv_results[key]['val_rmse']) < best_val_rmse:
        best_val_rmse = cv_results[key]['val_rmse']
    print('{:>8} - Train RMSE: {:.2f}, Val RMSE: {:.2f}'.format(key,
                                                     cv_results[key]['train_rmse'], 
                                                     cv_results[key]['val_rmse']))
print('\nBest validation RMSE is {:.2f}'.format(best_val_rmse))

# Lasso
from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold


param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

cv_results = dict()

for alpha in (0.001, 0.01, 0.1, 1, 10, 100):
    ridge = Lasso(alpha=alpha)
    model_result = model_eval(ridge, train_weather_df, val_weather_df, verbose=False)
    cv_results[alpha] = model_result


best_val_rmse = 100    
for key, value in cv_results.items():
    if (cv_results[key]['val_rmse']) < best_val_rmse:
        best_val_rmse = cv_results[key]['val_rmse']
    print('{:>8} - Train RMSE: {:.2f}, Val RMSE: {:.2f}'.format(key,
                                                     cv_results[key]['train_rmse'], 
                                                     cv_results[key]['val_rmse']))
print('\nBest validation RMSE is {:.2f}'.format(best_val_rmse))

# Adaboost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold

param_grid = {'n_estimators': [10, 50, 100, 400],
              'loss' : ['linear', 'square', 'exponential']}

cv_results = dict()

best_val_train = 100  
best_val_rmse = 100  
best_model = None

for n in [10, 50, 100, 400]:
    for loss in ['linear', 'square', 'exponential']:
        model = AdaBoostRegressor(n_estimators=n, loss=loss)
        model_result = model_eval(model, train_weather_df, val_weather_df, verbose=False)
        cv_results[(n, loss)] = model_result
        
        if model_result['val_rmse'] < best_val_rmse:
            best_result = model_result
            best_model = model

print('\nBest results: Train RMSE: {.2f}, Val RMSE is {:.2f}'.format(best_val_rmse))


print(best_model)

# Random Forests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold

num_feats = X_train.shape[1]
param_grid = {'n_estimators': [10, 50, 100, 400],
              'max_features' : ['auto', 'sqrt', 'log2'],
              'learning_rate' : [0.5, 0.75, 1.0]
              }

cv_results = dict()

best_val_rmse = 100  
best_model = None

for n_estimators in [10, 50, 100, 400, 800]:
    for max_features in ['auto', 'sqrt', 'log2']:
        for learning_rate in [0.5, 0.75, 1.0]:
            print('N = {}, max_features = {}, learning_rate = {}'.format(n_estimators, max_features, learning_rate))
            model = GradientBoostingRegressor(n_estimators=n, max_features=max_features, learning_rate=learning_rate)
            model_result = model_eval(model, train_weather_df, val_weather_df, verbose=False)
            cv_results[(n, loss)] = model_result

            if model_result['val_rmse'] < best_val_rmse:
                best_val_rmse = model_result['val_rmse']
                best_model = model

print('\nBest model: {}, validation RMSE is {:.2f}'.format(best_model, best_val_rmse))

# xgboost
import xgboost as xgb

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost

cv_results = dict()

for max_depth in (4, 6):
    for learning_rate in (0.01, 0.1, 0.3):
        for n_estimators in (200, 400, 800, 1200):
            for gamma in (0.0, 0.1, 0.2, 0.5, 1.0):
                for min_child_weight in (1, 3, 5):
                    for subsample in (0.8,):
                        for colsample_bytree in (0.8,):
                            print('Training XGB: {:>5}, {:>5}, {:>5} ,{:>5} ,{:>5} ,{:>5} ,{:>5}'.format(max_depth, learning_rate, n_estimators,
                                       gamma, min_child_weight, subsample, colsample_bytree), end='')

                            xgboost = xgb.XGBRegressor(objective="reg:linear",
                                                       max_depth=max_depth, 
                                                       learning_rate=learning_rate,
                                                       n_estimators=n_estimators,
                                                       gamma=gamma,
                                                       min_child_weight=min_child_weight,
                                                       subsample=subsample,
                                                       colsample_bytree=colsample_bytree,
                                                       silent=False,
                                                       seed=1234)

                            model_result = model_eval(xgboost, train_weather_df, val_weather_df, verbose=False)
                            cv_results[(max_depth, learning_rate, n_estimators,
                                       gamma, min_child_weight, subsample, colsample_bytree)] = (model_result['train_rmse'],
                                                                                                 model_result['val_rmse'])
                            print(', Train RMSE = {:.2f}, Val RMSE = {:.2f}'.format(model_result['train_rmse'], model_result['val_rmse']))                                                          


best_val_rmse = 100    
for key, value in cv_results.items():
    if (cv_results[key]['val_rmse']) < best_val_rmse:
        best_val_rmse = cv_results[key]['val_rmse']
    print('{:>8} - Train RMSE: {:.2f}, Val RMSE: {:.2f}'.format(key,
                                                     cv_results[key]['train_rmse'], 
                                                     cv_results[key]['val_rmse']))
print('\nBest validation RMSE is {:.2f}'.format(best_val_rmse))
print(model_result)

best_val_rmse = 100    
best_val_train = 100    

for key, value in cv_results.items():
    if cv_results[key][1] < best_val_rmse:
        best_train_rmse = cv_results[key][0]
        best_val_rmse = cv_results[key][1]
        best_params = key
        
#     print('{:>8} - Train RMSE: {:.2f}, Val RMSE: {:.2f}'.format(key,
#                                                      cv_results[key][0], 
#                                                      cv_results[key][1]))
print('\nBest params: {}, Train RMSE: {:.2f}, Val RMSE is {:.2f}'.format(best_params, best_train_rmse, best_val_rmse))
print(model_result)


print(cv_results)

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

num_feats = X_train.shape[1]
param_grid = {'C': [1, 10, 100],
              'kernel' : ['linear', 'rbf', 'poly'],
              'degree' : [3, 5],
              'gamma'  : [1e-3]
              }



cv_results = dict()

best_val_rmse = 100  
best_model = None

for C in [1, 10, 100]:
    for kernel in ['linear', 'rbf', 'poly']:
        for degree in  [3, 5]:
            for gamma in (1e-3,):
                print('C = {}, kernel = {}, degree = {}'.format(C, kernel, degree))
                model = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)
                model_result = model_eval(model, train_weather_df, val_weather_df, verbose=False)
                cv_results[(C, kernel, degree, gamma)] = model_result

                if model_result['val_rmse'] < best_val_rmse:
                    best_val_rmse = model_result['val_rmse']
                    best_model = model

print('\nBest model: {}, validation RMSE is {:.2f}'.format(best_model, best_val_rmse))

# Keras !!

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

NB_EPOCH=40
BATCH=256

# Create numpy training and validation arrays
print('Creating train and validation arrays')
X_train, y_train, _ = reg_x_y_split(train_weather_df, target_col='count',  ohe_cols=['day-hour'], verbose=False)
X_val, y_val, _ = reg_x_y_split(val_weather_df, target_col='count',  ohe_cols=['day-hour'], verbose=False)
n_train, n_feat = X_train.shape
n_val = X_val.shape[0]
print('Train dimensions {}, Validation dimensions {}'.format(X_train.shape, X_val.shape))

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)

model = Sequential()
model.add(Dense(2000, input_dim=n_feat, init='uniform'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1, init='uniform'))
# print('Model summary:\n')
# model.summary()

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd)

print('\nTraining model\n')
model.fit(X_train_std, y_train,
          nb_epoch=NB_EPOCH,
          batch_size=BATCH,
          verbose=2)

print('\nGenerating predictions on test set\n')
val_rmse = np.sqrt(model.evaluate(X_val_std, y_val, batch_size=BATCH))

print('\nValidation error {:.4f}'.format(val_rmse))

y_val_pred = model.predict(X_val_std).squeeze()
# val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_all_results(result_val_df, 'true', 'pred', 'Keras validation dataset prediction')



y_val_pred.shape



