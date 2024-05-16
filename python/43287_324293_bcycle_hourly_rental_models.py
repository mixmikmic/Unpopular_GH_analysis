import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

import datetime

from bcycle_lib.utils import *

get_ipython().magic('matplotlib inline')
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

bikes_df = load_bikes()
stations_df = load_stations()
weather_df = load_weather()

# Convert the long-form data into wide form for pandas aggregation by hour
hourly_df = load_bike_trips()
hourly_df = hourly_df.reset_index()

hourly_df = hourly_df.pivot_table(index='datetime', values='checkouts', columns='station_id')
hourly_df = hourly_df.resample('1H').sum()
hourly_df = hourly_df.sum(axis=1)
hourly_df = pd.DataFrame(hourly_df, columns=['rentals'])
hourly_df = hourly_df.fillna(0)
hourly_df.head()

# Plot out the hourly bike rentals
def plot_rentals(df, cols, title, times=None):
    ''' Plots a time series data'''
    
    fig, ax = plt.subplots(1,1, figsize=(20,6))

    if times is not None:
        ax = df[times[0]:times[1]].plot(y=cols, ax=ax) # , color='black', style=['--', '-'])
        title += ' ({} to {})'.format(times[0], times[1])
    else:
        ax = df.plot(y=cols, ax=ax) # , color='black', style=['--', '-'])

    ax.set_xlabel('Date', fontdict={'size' : 14})
    ax.set_ylabel('Rentals', fontdict={'size' : 14})
    ax.set_title(title, fontdict={'size' : 16}) 
    ttl = ax.title
    ttl.set_position([.5, 1.02])
#     ax.legend(['Hourly rentals'], fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)   

plot_rentals(hourly_df, 'rentals', 'Hourly aggregated rentals', ('2016-04-01', '2016-04-08'))

# Wow - that looks spiky ! Let's smooth this out
def smooth_ts(df, halflife):
    '''Smooths time series data using ewma and halflife
    INPUT: Dataframe to smooth, halflife for Exponential Weighted Moving Average
    RETURNS: Smoothed dataframe
    '''
    smooth_df = df.ewm(halflife=halflife, ignore_na=False,adjust=True,min_periods=0).mean()
    smooth_df = smooth_df.shift(periods=-halflife)
    smooth_df = smooth_df.fillna(0)
    return smooth_df

# plot_df['original'] = hourly_df['rentals']
# plot_rentals(hourly_smooth_df, ['rentals', 'Hourly aggregated rentals', ('2016-04-01', '2016-04-08'))

smooth_df = smooth_ts(hourly_df, 2)

# First create a daily rentals dataframe, split it into training and validation

def add_time_features(df):
    ''' Extracts dayofweek and hour fields from index
    INPUT: Dataframe to extract fields from
    RETURNS: Dataframe with dayofweek and hour columns
    
    '''
    df.head()
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour.astype(str)
    df['day-hour'] = df['dayofweek'].astype(str) + '-' + df['hour']
    df['weekday'] = (df['dayofweek'] < 5).astype(np.uint8)
    df['weekend'] = (df['dayofweek'] >= 5).astype(np.uint8)
    return df

def split_train_val_df(df, train_start, train_end, val_start, val_end):
    '''Splits Dataframe into training and validation datasets
    INPUT: df - Dataframe to split
           train_start/end - training set time range
           val_start/end - validation time range
    RETURNS: Tuple of (train_df, val_df)
    '''
    train_df = df.loc[train_start:train_end,:]
    val_df = df.loc[val_start:val_end,:]
    return (train_df, val_df)

rental_time_df = add_time_features(hourly_df)
train_df, val_df = split_train_val_df(rental_time_df, 
                                      '2016-04-01', '2016-05-15', 
                                      '2016-05-16', '2016-05-31')


print('\nTraining data first and last row:\n{}\n{}'.format(train_df.iloc[0], train_df.iloc[-1]))
print('\nValidation data first and last row:\n{}\n{}'.format(val_df.iloc[0], val_df.iloc[-1]))

val_df.head()

def RMSE(pred, true):
    '''
    Calculates Root-Mean-Square-Error using predicted and true
    columns of pandas dataframe
    INPUT: pred and true pandas columns
    RETURNS: float of RMSE
    '''
    rmse = np.sqrt(np.sum((pred - true).apply(np.square)) / pred.shape[0])
    return rmse

def plot_val(val_df, pred_col, true_col, title):
    '''
    Plots the validation prediction
    INPUT: val_df - Validation dataframe
           pred_col - string with prediction column name
           true_col - string with actual column name
           title - Prefix for the plot titles.
    RETURNS: Nothing
    '''
    def plot_ts(df, pred, true, title, ax):
        '''Generates one of the subplots to show time series'''
        ax = df.plot(y=[pred, true], ax=ax) # , color='black', style=['--', '-'])
        ax.set_xlabel('Date', fontdict={'size' : 14})
        ax.set_ylabel('Rentals', fontdict={'size' : 14})
        ax.set_title(title, fontdict={'size' : 16}) 
        ttl = ax.title
        ttl.set_position([.5, 1.02])
        ax.legend(['Predicted rentals', 'Actual rentals'], fontsize=14, loc=2)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
    
    fig, ax = plt.subplots(1,1, sharey=True, figsize=(16,8))
    plot_ts(val_df, pred_col, true_col, title + ' (validation set)', ax)
    

def plot_prediction(train_df, val_df, pred_col, true_col, title):
    '''
    Plots the predicted rentals along with actual rentals for the dataframe
    INPUT: train_df, val_df - pandas dataframe with training and validataion results
           pred_col - string with prediction column name
           true_col - string with actual column name
           title - Prefix for the plot titles.
    RETURNS: Nothing
    '''
    def plot_ts(df, pred, true, title, ax):
        '''Generates one of the subplots to show time series'''
        ax = df.plot(y=[pred, true], ax=ax) # , color='black', style=['--', '-'])
        ax.set_xlabel('Date', fontdict={'size' : 14})
        ax.set_ylabel('Rentals', fontdict={'size' : 14})
        ax.set_title(title, fontdict={'size' : 16}) 
        ttl = ax.title
        ttl.set_position([.5, 1.02])
        ax.legend(['Predicted rentals', 'Actual rentals'], fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)   
    
    fig, axes = plt.subplots(2,1, sharey=True, figsize=(20,12))
    plot_ts(train_df, pred_col, true_col, title + ' (training set)', axes[0])
    plot_ts(val_df, pred_col, true_col, title + ' (validation set)', axes[1])
    
def plot_residuals(train_df, val_df, pred_col, true_col, title):
    '''
    Plots the residual errors in histogram (between actual and prediction)
    INPUT: train_df, val_df - pandas dataframe with training and validataion results
           pred_col - string with prediction column name
           true_col - string with actual column name
           title - Prefix for the plot titles.
    RETURNS: Nothing

    '''
    def plot_res(df, pred, true, title, ax):
        '''Generates one of the subplots to show time series'''
        residuals = df[pred] - df[true]
        ax = residuals.plot.hist(ax=ax, bins=20)
        ax.set_xlabel('Residual errors', fontdict={'size' : 14})
        ax.set_ylabel('Count', fontdict={'size' : 14})
        ax.set_title(title, fontdict={'size' : 16}) 
        ttl = ax.title
        ttl.set_position([.5, 1.02])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)   
    
    fig, axes = plt.subplots(1,2, sharey=True, sharex=True, figsize=(20,6))
    plot_res(train_df, pred_col, true_col, title + ' residuals (training set)', axes[0])
    plot_res(val_df, pred_col, true_col, title + ' residuals (validation set)', axes[1])
    
    
def plot_results(train_df, val_df, pred_col, true_col, title):
    '''Plots time-series predictions and residuals'''
    plot_prediction(train_df, val_df, pred_col, true_col, title=title)
    plot_residuals(train_df, val_df, pred_col, true_col, title=title)
    
def plot_scores(df, title, sort_col=None):
    '''Plots model scores in a horizontal bar chart
    INPUT: df - dataframe containing train_rmse and val_rmse columns
           sort_col - Column to sort bars on
    RETURNS: Nothing
    '''
    fig, ax = plt.subplots(1,1, figsize=(12,8)) 
    if sort_col is not None:
        scores_df.sort_values(sort_col).plot.barh(ax=ax)
    else:
        scores_df.sort_values(sort_col).plot.barh(ax=ax)

    ax.set_xlabel('RMSE', fontdict={'size' : 14})
    ax.set_title(title, fontdict={'size' : 18}) 
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.legend(['Train RMSE', 'Validation RMSE'], fontsize=14, loc=0)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

# Create a dataframe with median hourly rentals by day. Has 7 x 24 = 168 rows
avg_df = train_df.groupby(['hour']).median().reset_index()

train_avg_df = pd.merge(train_df, avg_df, 
                        on='hour',
                        suffixes=('_true', '_pred'),
                        how='left')
train_avg_df = train_avg_df.rename(columns ={'rentals_true' : 'true', 'rentals_pred' : 'pred'})
train_avg_df = train_avg_df[['hour', 'true', 'pred']]
train_avg_df.index = train_df.index

val_avg_df = pd.merge(val_df, avg_df, 
                      on='hour',
                      suffixes=('_true', '_pred'),
                      how='left')
val_avg_df = val_avg_df.rename(columns ={'rentals_true' : 'true', 'rentals_pred' : 'pred'})
val_avg_df = val_avg_df[['hour', 'true', 'pred']]
val_avg_df.index = val_df.index

train_avg_df.head(8)

# Store the results of the median RMSE and plot prediction
train_avg_rmse = RMSE(train_avg_df['pred'], train_avg_df['true'])
val_avg_rmse = RMSE(val_avg_df['pred'], val_avg_df['true'])

# Store the evaluation results
scores_df = pd.DataFrame({'train_rmse' : train_avg_rmse, 'val_rmse' : val_avg_rmse}, index=['hourly_median'])

# Print out the RMSE metrics and the prediction
print('Hourly Median Baseline RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_avg_rmse, val_avg_rmse))
# plot_results(train_avg_df, val_avg_df, 'pred', 'true', title='Average baseline')
plot_val(val_avg_df, 'pred', 'true', title='Hourly median baseline prediction')

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, scale

def reg_x_y_split(df, target_col, ohe_cols=None, z_norm_cols=None, minmax_norm_cols=None):
    ''' Returns X and y to train regressor
    INPUT: df = Dataframe to be converted to numpy arrays 
           target_col = Column name of the target variable
           ohe_col = Categorical columns to be converted to one-hot-encoding
           z_norm_col = Columns to be z-normalized
    RETURNS: Tuple with X, y, df
    '''
    
    # Create a copy, remove index and date fields
    df_out = df.copy()
    df_X = df.copy()
    df_X = df_X.reset_index(drop=True)
    X = None
    
    # Convert categorical columns to one-hot encoding
    if ohe_cols is not None:
        for col in ohe_cols:
            print('Binarizing column {}'.format(col))
            lbe = LabelBinarizer()
            ohe_out = lbe.fit_transform(df_X[col])
            if X is None:
                X = ohe_out
            else:
                X = np.hstack((X, ohe_out))
            df_X = df_X.drop(col, axis=1)
            
    # Z-normalize relevant columns
    if z_norm_cols is not None:
        for col in z_norm_cols:
            print('Z-Normalizing column {}'.format(col))
            scaled_col = scale(df[col].astype(np.float64))
            scaled_col = scaled_col[:,np.newaxis]
            df_out[col] = scaled_col
            if X is None:
                X = scaled_col
            if X is not None:
                X = np.hstack((X, scaled_col))
            df_X = df_X.drop(col, axis=1)

    if minmax_norm_cols is not None:
        for col in minmax_norm_cols:
            print('Min-max scaling column {}'.format(col))
            mms = MinMaxScaler()
            mms_col = mms.fit_transform(df_X[col])
            mms_col = mms_col[:, np.newaxis]
            df_out[col] = mms_col
            if X is None:
                X = mms_col
            else:
                X = np.hstack((X, mms_col))
            df_X = df_X.drop(col, axis=1)

    # Combine raw pandas Dataframe with encoded / normalized np arrays
    if X is not None:
        X = np.hstack((X, df_X.drop(target_col, axis=1).values))
    else:
        X = df_X.drop(target_col, axis=1)
        
    y = df[target_col].values

    return X, y, df_out

# Create new time-based features, numpy arrays to train model

X_train, y_train, train_df = reg_x_y_split(train_df, 
                                           target_col='rentals', 
                                           ohe_cols=['day-hour'])
X_val, y_val, val_df = reg_x_y_split(val_df, 
                                     target_col='rentals', 
                                     ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

# Linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
reg_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
reg_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : reg_train_rmse, 'val_rmse' : reg_val_rmse}, 
                                              index=['linreg_time']))

def df_from_results(index_train, y_train, y_train_pred, index_val, y_val, y_val_pred):
    
    train_dict = dict()
    val_dict = dict()

    train_dict['true'] = y_train
    train_dict['pred'] = y_train_pred

    val_dict['true'] = y_val
    val_dict['pred'] = y_val_pred

    train_df = pd.DataFrame(train_dict)
    val_df = pd.DataFrame(val_dict)

    train_df.index = index_train
    val_df.index = index_val
    
    return train_df, val_df
    
    
reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

print('Time regression RMSE - Train: {:.2f}, Val: {:.2f}'.format(reg_train_rmse, reg_val_rmse))
# plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Time regression')
plot_val(reg_result_val_df, 'pred', 'true', title='Time regression prediction')

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
    df = df.set_index('datetime')
    
    # Extract the date field to join on
    weather_df = weather_df.reset_index()
    
    # Merge with the weather information using the date
    merged_df = pd.merge(df, weather_df, on='date', how='left')
    merged_df.index = df.index
    merged_df = merged_df.drop('date', axis=1)
    
    assert df.shape[0] == merged_df.shape[0], "Error - row mismatch after merge"
    
    return merged_df


train_weather_df = merge_daily_weather(train_df, weather_df)
val_weather_df = merge_daily_weather(val_df, weather_df)

train_weather_df.head()

X_train, y_train, _ = reg_x_y_split(train_weather_df, target_col='rentals', ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_weather_df, target_col='rentals', ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
reg_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
reg_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_weather' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : reg_train_rmse, 'val_rmse' : reg_val_rmse}, 
                                              index=['linreg_time_weather']))

print('Time and weather RMSE - Train: {:.2f}, Val: {:.2f}'.format(reg_train_rmse, reg_val_rmse))

reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

# plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Linear regression with weather')
plot_val(reg_result_val_df, 'pred', 'true', title='Time and weather regression prediction')

# print('Regression coefficients:\n{}'.format(reg.coef_))
# print('Regression residues:\n{}'.format(reg.residues_))
# print('Regression intercept:\n{}'.format(reg.intercept_))
scores_df.sort_values('val_rmse', ascending=True).plot.barh()

corr_df = train_weather_df.corr()

fig, ax = plt.subplots(1,1, figsize=(12, 12))
sns.heatmap(data=corr_df, square=True, linewidth=2, linecolor='white', ax=ax)
ax.set_title('Weather dataset correlation', fontdict={'size' : 18})
ttl = ax.title
ttl.set_position([.5, 1.05])
# ax.set_xlabel('Week ending (Sunday)', fontdict={'size' : 14})
# ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)

ohe_cols = ['day-hour']
znorm_cols = ['max_temp', 'min_temp', 'max_humidity', 'min_humidity',
       'max_pressure', 'min_pressure', 'max_wind', 'min_wind', 'max_gust', 'precipitation']
minmax_cols = ['cloud_pct']
target_col = 'rentals'

X_train, y_train, train_df = reg_x_y_split(train_weather_df, target_col, ohe_cols, znorm_cols, minmax_cols)
X_val, y_val, val_df = reg_x_y_split(val_weather_df, target_col, ohe_cols, znorm_cols, minmax_cols)

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
reg_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
reg_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# # Store the evaluation results
# if 'linreg_time_weather_norm' not in scores_df.index:
#     scores_df = scores_df.append(pd.DataFrame({'train_rmse' : reg_train_rmse, 'val_rmse' : reg_val_rmse}, 
#                                               index=['linreg_time_weather_norm']))

print('Time and Weather Regression RMSE - Train: {:.2f}, Val: {:.2f}'.format(reg_train_rmse, reg_val_rmse))

reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

# plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Linear regression with time and weather')
plot_val(reg_result_val_df, 'pred', 'true', title='Linear regression with time and weather normalization')

# Try different values for the good columns
GOOD_COLS = ['rentals', 'max_temp', 'min_temp', 'max_gust', 'precipitation', 
        'cloud_pct', 'thunderstorm', 'day-hour']

X_train, y_train, train_df = reg_x_y_split(train_weather_df[GOOD_COLS], target_col='rentals', 
                                           ohe_cols=['day-hour'],
#                                            z_norm_cols=['max_temp', 'min_temp', 'max_gust'],
                                           minmax_norm_cols= ['cloud_pct'])

X_val, y_val, val_df = reg_x_y_split(val_weather_df[GOOD_COLS], target_col='rentals', 
                                     ohe_cols=['day-hour'],
#                                      z_norm_cols=['max_temp', 'min_temp', 'max_gust'],
                                     minmax_norm_cols=['cloud_pct'])

print('train_df columns: {}'.format(train_df.columns))
print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_weather_feat' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
                                              index=['linreg_time_weather_feat']))

print('Time and Weather Feature Regression RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

# plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Time and Weather Feature Regression')
plot_val(reg_result_val_df, 'pred', 'true', title='Time and Weather Feature Regression')

from sklearn.linear_model import Ridge

alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
ridge_cv_scores = dict()

for alpha in alphas:
    reg = Ridge(alpha=alpha, max_iter=10000)
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_val_pred = reg.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    ridge_cv_scores[alpha] = (train_rmse, val_rmse)


ridge_cv_df = pd.DataFrame(ridge_cv_scores).transpose().reset_index()
ridge_cv_df.columns = ['alpha', 'train_rmse', 'val_rmse']
ridge_cv_df.plot.line(x='alpha', y=['train_rmse', 'val_rmse'], logx=True)

# Store the evaluation results
if 'ridge_cv' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : ridge_cv_df['train_rmse'].min(), 
                                               'val_rmse' : ridge_cv_df['val_rmse'].min()}, 
                                              index=['ridge_cv']))
    
ridge_cv_df

from sklearn.linear_model import Lasso

alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
ridge_cv_scores = dict()

for alpha in alphas:
    reg = Lasso(alpha=alpha, max_iter=10000)
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_val_pred = reg.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    ridge_cv_scores[alpha] = (train_rmse, val_rmse)

lasso_cv_df = pd.DataFrame(ridge_cv_scores).transpose().reset_index()
lasso_cv_df.columns = ['alpha', 'train_rmse', 'val_rmse']
lasso_cv_df.plot.line(x='alpha', y=['train_rmse', 'val_rmse'], logx=True)

# Store the evaluation results
if 'lasso_cv' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : ridge_cv_df['train_rmse'].min(), 
                                               'val_rmse' : ridge_cv_df['val_rmse'].min()}, 
                                              index=['lasso_cv']))
    
lasso_cv_df

# Now plot side-by-side comparisons of the hyperparameter tuning, with the OLS result as a horizontal line
def plot_cv(df, title, ax):
    '''Generates one of the subplots to show time series'''
    df.plot.line(x='alpha', y=['train_rmse', 'val_rmse'], logx=True, ax=ax)
    ax.set_xlabel('Alpha', fontdict={'size' : 14})
    ax.set_ylabel('RMSE', fontdict={'size' : 14})
    ax.set_title(title, fontdict={'size' : 18}) 
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)   
    ax.axhline(y=scores_df.loc['linreg_time_weather_feat', 'val_rmse'], color='g', linestyle='dashed')
    ax.axhline(y=scores_df.loc['linreg_time_weather_feat', 'train_rmse'], color='b', linestyle='dashed')


    ax.legend(['Train RMSE', 'Validation RMSE'], fontsize=14, loc=0)


    
fig, axes = plt.subplots(1,2, sharey=True, figsize=(20,6))
plot_cv(lasso_cv_df, 'Lasso regression alpha', ax=axes[0])
plot_cv(ridge_cv_df, 'Ridge regression alpha', ax=axes[1])
    

plot_scores(scores_df, 'Model scores', 'val_rmse')
scores_df.round(2)

# Random forest

from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(max_depth=100, min_samples_split=40)
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# # Store the evaluation results
# if 'linreg_time_weather_norm_feat' not in scores_df.index:
#     scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
#                                               index=['linreg_time_weather_norm_feat']))

print('Weather Feature Regression RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Linear regression with weather')
# plot_prediction(reg_result_val_df, 'pred', 'true', title='Linear Regression with Weather Validation set prediction')

sns.jointplot(data=train_weather_df, x='precipitation', y='rentals', size=10, kind='reg')

train_weather_df['prec_square'] = train_weather_df['precipitation'].apply(lambda x: np.power(x, 0.2))
sns.jointplot(data=train_weather_df, x='prec_square', y='rentals', size=10, kind='reg')



