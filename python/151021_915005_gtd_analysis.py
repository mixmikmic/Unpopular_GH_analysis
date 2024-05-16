# Dependencies

import math, os
import numpy as np
import pandas as pd
import random
import time

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

params = {'legend.fontsize': 'xx-small',
         'axes.labelsize': 'xx-small',
         'axes.titlesize':'xx-small',
         'xtick.labelsize':'xx-small',
         'ytick.labelsize':'xx-small'}
plt.rcParams.update(params)

# Importing the data as pandas DataFrame

df = pd.read_excel("gtd_95to12_0617dist.xlsx", sheetname=0)

# renaming date time to change type into timestamp later
df = df.rename(columns={'iyear': 'year', 'imonth': 'month', 'iday': 'day'})

# word_tokenize(str(df['summary'].loc[55051].split()))
df['day'][df.day == 0] = 1
df['date'] = pd.to_datetime(df[['day', 'month', 'year']])

df.head()

df.columns

# missing values: NaN values in the data

def missing_values():
    temp_dict = dict()
    for i in df.columns:
        if df[i].isnull().sum() > 0: 
            temp_dict[i] = df[i].isnull().sum()
    return temp_dict

# gives the information of missing values in each of the decorations
len(missing_values())

# First DROPPING the columns that contain more than 50% of its data NaNs

def delete_columns(col):
    if df[col].isnull().sum() > df[col].count()/2:
        del df[col]

for col in df.columns:
    delete_columns(col)

df.head(n=5)

# these are the columns left with us now
df.columns, print(df.columns.shape)

imp_columns = {'eventid', 'year', 'month', 'day', 'date', 'country', 'country_txt', 'region_txt', 'provstate', 'city', 
              'latitude', 'longitude', 'summary', 'crit1', 'success', 'suicide', 'attacktype1_txt', 'targtype1_txt', 
              'gname', 'motive', 'claimed', 'weaptype1_txt', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 
              'nwoundte'}

data_frame = pd.DataFrame(df, columns=imp_columns)
data_frame.head()

data_frame[data_frame.crit1==0].shape

# if crit1 == 0: then the events recorded might not be the actual terrorist activity

data_frame[(data_frame.crit1 == 0) & ((data_frame.gname =='Gunmen')|(data_frame.gname == 'Unknown'))].shape

# Now we are left with these NaNs
# We need to engineer these missing values if we want to use these columns for 
# prediction model development later

missing_values()

data_frame['claimed'].unique()

data_frame['claimed'] = data_frame['claimed'].fillna(0)

data_frame['claimed'].unique()

data_frame[data_frame['city'].isnull()].shape[0], data_frame[data_frame['longitude'].isnull()].shape[0]

tmp_list = ['country_txt', 'region_txt', 'city', 'longitude', 'latitude']

data_frame[tmp_list][data_frame['latitude'].isnull() & data_frame['city'].isnull()].shape

rows_missing_both_city_coords = data_frame[['city', 'latitude', 'longitude']][data_frame['latitude'].isnull() & 
                                data_frame['city'].isnull()].index.tolist()

# the new df ahead will not contain those 17 events: dropping those events 
data_frame = data_frame.drop(rows_missing_both_city_coords)

all_cities = data_frame['city'].unique().tolist()
all_cities[:10]

data_frame[['city', 'longitude', 'latitude']].head()

data_frame.shape

# storing the mode 'latitude' and 'longitude' values of each city: i.e. the most frequent value of the coordinates
# from the known values corresponding to the events occuring in the same city

city_coords_dict = dict()

for city in all_cities:
    long = float(data_frame['longitude'][data_frame['city'] == city].median())
    lat = float(data_frame['latitude'][data_frame['city'] == city].median())
    city_coords_dict[city] = [long, lat]

# equal number of events missing latitude and longitude => same events missing both?

print(data_frame[data_frame['longitude'].isnull()==True]['country_txt'].shape)
print(data_frame[data_frame['latitude'].isnull()==True]['country_txt'].shape)

# Import the basemap package
from mpl_toolkits.basemap import Basemap
from IPython.display import set_matplotlib_formats
from mpl_toolkits import basemap

# Turn on retina display mode
set_matplotlib_formats('retina')
# turn off interactive mode
plt.ioff()

# printing the all possible projections for the earth(sphere) to 2D plane map projections
print(basemap.supported_projections)

# Plotting the incidents locations in the map

fig = plt.figure(figsize=(20,15))
m = Basemap(projection='robin', lon_0=0)

# background color of the map - greyish color
fig.patch.set_facecolor('#e6e8ec')

#m.drawmapboundary(fill_color='aqua')

# Draw coastlines, and the edges of the map.
m.drawcoastlines(color='black')
m.fillcontinents(color='burlywood', lake_color='lightblue')
m.drawcountries(linewidth=1, linestyle='dashed' ,color='black')
m.drawmapboundary(fill_color='lightblue')

graticule_width = 20
graticule_color = 'white'

m.drawmapboundary()

# Convert latitude and longitude to x and y coordinates
xs = list(data_frame['longitude'].astype(float))
ys = list(data_frame['latitude'].astype(float))

num_killed = list(data_frame['nkill'].astype(float))

x, y = m(xs, ys)
m.plot(x, y, "o", markersize=5, alpha=1)
plt.show()

data_frame['weaptype1_txt'].unique()

data_frame['attacktype1_txt'].unique()

data_frame['targtype1_txt'].unique().shape

# In Nepal
in_nepal = data_frame.loc[data_frame['country_txt']=='Nepal']
# Total number of people killed in Nepal
in_nepal['nkill'].sum()

countries_list = df['country_txt'].unique().tolist()
countries_list
country_killed = dict()

for country in countries_list:
    tmp_df = df.loc[df['country_txt']==country]
    num_killed = tmp_df['nkill'].sum()
    country_killed[country] = num_killed

sorted_country_killed_dict = sorted(country_killed.items(), key= lambda x: x[1], reverse=True)
num_killed_list = [x[1] for x in sorted_country_killed_dict if x[1] > 1500]
corresponding_countries_list = [x[0] for x in sorted_country_killed_dict if x[1] > 1500]

# country name and number of people killed in each country
list(filter(lambda x: x[1] > 1500, sorted_country_killed_dict))

plt.figure(figsize=(10,10))
plt.pie(num_killed_list, labels=corresponding_countries_list, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.title('COUNTRYWISE KILLING', fontsize=20)
plt.show()

tmp_df = pd.DataFrame(data_frame, columns={'eventid', 'year', 'country', 'longitude', 'latitude', 'claimed', 'suicide',
                                   'attacktype1', 'weaptype1'})
fig, ax = plt.subplots(figsize=(12,8))
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 1.0})
ax = sns.heatmap(tmp_df.corr(), cbar=True, ax=ax)
plt.show()

# Total number of incidents without claim: not 0 or 1 value, something else

original_length = len(df)
length = len(df.loc[df['claimed'].isin({0,1})])
print("Without claim incidents: ", original_length - length)

# Plotting the bar diagrams of countries with most incidents
country_list = data_frame['country_txt']
incident_country = country_list.value_counts()

my_dict = dict(incident_country)
new_dict = {}
for key in my_dict.keys():
    if my_dict[key] > 1000:
        new_dict.update({key : my_dict[key]})

countries_with_high_incidents = list(new_dict.keys())
num_incidents = list(new_dict.values())
my_list = []
for i in range(len(countries_with_high_incidents)):
    for j in range(num_incidents[i]):
        my_list.append(countries_with_high_incidents[i])

# Plotting the number of incidents in countries with most incidents
fig, ax = plt.subplots(figsize=(40,20)) 
sns.set(style='darkgrid')
sns.set_context("notebook", font_scale=4, rc={"lines.linewidth": 1.5})
ax = sns.countplot(my_list)
plt.show()

data_frame.columns

# reindexing the data_frame
data_frame.index = range(len(data_frame))
# filling NaN in nkill by 0
data_frame['nkill'] = data_frame['nkill'].fillna(0)

years = data_frame['year'].unique()
# create a dictionary to store year and numbers of people killed in the year
year_killed = dict()
# initializing the dictionary
for year in years:
    year_killed[str(year)] = 0 

# running through the years
for year in years:
    for i in range(len(data_frame)):
        if data_frame['year'].loc[i] == year:
            year_killed[str(year)] += data_frame['nkill'].loc[i]

events_year = dict(data_frame['year'].value_counts())
tmp_df = pd.DataFrame()
tmp_df['year'] = events_year.keys()
tmp_df['events_num'] = events_year.values()
tmp_df.sort_values('year', ascending=True, inplace=True)

# adding number of people killed each year column in the tmp_df
tmp_df['killed'] = year_killed.values()
tmp_df.head()

# Create scatterplot of dataframe
sns.set()
plt.figure(figsize=(12,6))
ax = sns.lmplot('year', 'events_num', data=tmp_df, hue='killed', fit_reg=False, 
                scatter_kws={"marker": "0", "s": 150}, aspect=1.5)
plt.title('Number of events by Year', fontsize=16)
plt.ylabel('Number of Incidents', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()

# sklearn methods

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm

from sklearn.model_selection import cross_val_score

interesting_columns = {'country', 'longitude', 'latitude', 'claimed', 'suicide','attacktype1_txt',  
                      'targtype1_txt', 'weaptype1_txt', 'gname'}
train_df = pd.DataFrame(data_frame, columns=interesting_columns)
# removing all the NaN values if still anywhere in the data
train_df.dropna(inplace=True)
train_df.index = range(len(train_df))
# the preview of current dataframe
train_df.head()

y_train, y_index = pd.factorize(train_df['gname'])
y_train.shape, y_index.shape

# training X matrix - without target variable: 'gname'
x_train = pd.get_dummies(train_df.drop('gname', axis=1))
x_train.shape

# Data preparation for the model input : into list

x_train_array = np.array([x_train.iloc[i].tolist() for i in range(len(x_train))])
y_train_array = y_train

# Random Forest Classifier 

rfc = RandomForestClassifier()
rfc.fit(x_train_array[:52000], y_train_array[:52000])

rfc.predict(x_train_array[52000:])

predicted = rfc.predict(x_train_array[52000:])

rfc.score(x_train_array[52000:], y_train_array[52000:])

actual = list(train_df['gname'][52000:])
predicted_index = [y_index[i] for i in predicted]

# Let us look at few predictions and corresponding actual group names
for i in range(1000, 1010):
    print("Predicted: ", predicted_index[i], "\tand", "Actual: ", actual[i])

# Looking into the most important features in the train data

max_importance = rfc.feature_importances_.max()
print("Maximum feature importance value: ", max_importance)
print('')

print('feature', '\t', 'feature_importance')
for index, value in enumerate(rfc.feature_importances_):
    
    # printing the most important features
    if value > 0.01:
        print(x_train.columns[index], '\t', value)

# Mean accuracy of the prediction: 
# accuracy of predict(x_train_list[50000:]) with respect to y_train_list[50000:]

rfc.score(x_train_array[52000:], y_train_array[52000:])

scores = cross_val_score(rfc, x_train_array[52000:55000], y_train_array[52000:55000])
print(scores, 'and mean value: ', scores.mean())

rfc.decision_path(x_train_array[52000:])

clf = svm.SVC()
clf.fit(x_train_array[:52000], y_train_array[:52000])

clf.predict(x_train_array[52000:])

clf.score(x_train_array[52000:], y_train_array[52000:])

cross_val_scores_val_scores_val_score(clf, x_train_array[52000:], y_train_array[52000:])

clf.support_vectors_

clf.support_

clf.n_support_

print(__doc__)

