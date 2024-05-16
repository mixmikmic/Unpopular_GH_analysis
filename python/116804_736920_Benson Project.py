# imports a library 'pandas', names it as 'pd'
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

import datetime
from datetime import datetime as dt
import calendar

import pickle
from copy import deepcopy
from collections import defaultdict

# enables inline plots, without it plots don't show up in the notebook
get_ipython().magic('matplotlib inline')
get_ipython().magic('autosave 120')

print("Pandas version:",pd.__version__)
print("Numpy version:",np.__version__)

# various options in pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.precision', 3)

MTA_May13 = pd.read_csv('http://web.mta.info/developers/data/nyct/turnstile/turnstile_170513.txt')
MTA_May06 = pd.read_csv('http://web.mta.info/developers/data/nyct/turnstile/turnstile_170506.txt')
MTA_Apr29 = pd.read_csv('http://web.mta.info/developers/data/nyct/turnstile/turnstile_170429.txt')

MTAdata_orig = pd.concat([MTA_Apr29, MTA_May06, MTA_May13], ignore_index = True)

MTAdata_orig.rename(columns=lambda x: x.strip(), inplace=True) #rid of whitespace

pickle.dump(MTAdata_orig, open("data/MTAdata_orig", "wb"))
MTAdata_orig.head(10)

MTAdata = pickle.load(open("data/MTAdata_orig", "rb"))

MTAdata['date'] = MTAdata['DATE'].apply(lambda x: dt.strptime(x, '%m/%d/%Y')) #make datetime objects

MTAdata['week_day_num'] = MTAdata['date'].apply(lambda x: x.weekday()) #names of the days of week

MTAdata['week_day_name'] = MTAdata['date'].dt.weekday_name#numeric indicies that relate to day of week

MTAdata['time'] = MTAdata['TIME'].apply(lambda x: dt.strptime(x, '%H:%M:%S').time()) #makes time object

MTAdata['turnstile'] = MTAdata['C/A'] + " "+ MTAdata['SCP'] #identifies unique turnstile

MTAdata.head(10)

#turnstile level for entries
MTAdata['turnstile entries'] = MTAdata.groupby(['turnstile'])['ENTRIES'].transform(lambda x: x.diff())

#turnstile level for exits
MTAdata['turnstile exits'] = MTAdata.groupby(['turnstile'])['EXITS'].transform(lambda x: x.diff())

#the starting point of our differences will have NaN's, lets change those to 0's!
MTAdata['turnstile entries'].fillna(0, inplace=True)
MTAdata['turnstile entries'].fillna(0, inplace=True)

#Some of the entry and exit numbers are negative. We assume they are due to the counter errors from the original data. 
#The rows with negative entry or exit numbers will be removed.
MTAdata = MTAdata[(MTAdata['turnstile entries'] > 0) & (MTAdata['turnstile exits'] > 0)]
MTAdata

#the total traffic at a turnstile in a given block of time (4hrs) is the sum of entries and exits.
MTAdata['traffic'] = MTAdata['turnstile entries'] + MTAdata['turnstile exits']

#since our analysis is at the turnstile level, we assume an upper bound of 5000 individuals through a turnstile
#in a block of time.
MTAdata = MTAdata[MTAdata['traffic'] <= 5000]

#Save MTA dataframe locally.
pickle.dump(MTAdata, open("data/MTAdata_clean", "wb"))

#Load clean MTA dataframe.
MTAdata = pickle.load(open("data/MTAdata_clean", "rb"))

MTAdata.head()

average_traffic_per_day = int((sum(MTAdata['traffic'])/3)/7)
print("Average Traffic through turnstiles per day: ", average_traffic_per_day)

#Get the total traffic data for each day of the week.
day_total_traffic = pd.DataFrame(MTAdata.groupby(['week_day_name', 'week_day_num'])['traffic'].sum()).reset_index()

#Get the average traffic data for each day of the week.
day_average_traffic = deepcopy(day_total_traffic)
day_average_traffic['traffic'] = day_average_traffic['traffic'] / 3
day_average_traffic = day_average_traffic.sort_values('week_day_num')

#plot the Average Daily Traffic on Day of Week
fig, ax = plt.subplots()
fig.set_size_inches(10,6)
rc={'axes.labelsize': 16, 'font.size': 16, 'legend.fontsize': 32.0, 'axes.titlesize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
sns.set(rc = rc)
sns.barplot(x = day_average_traffic['week_day_name'], y = day_average_traffic['traffic'])
ax.set_xlabel('Day of Week')
ax.set_ylabel('Total Turnstile Traffic');
#fig.savefig("images/Traffic on Day of Week for NYC MTA System.png")

#Get the average daily traffic data for each station based on weekday traffic.
station_day_total_traffic = pd.DataFrame(MTAdata.groupby(['STATION'])['traffic'].sum()).reset_index()
station_day_average_traffic = deepcopy(station_day_total_traffic)
station_day_average_traffic['traffic'] = station_day_total_traffic['traffic'] / 21
station_day_average_traffic.head(5)

#plot the average daily traffic data for top stations based on weekday traffic
top_5_list = [('TIMES SQ-42 ST'), ('GRD CNTRL-42 ST'), ('34 ST-HERALD SQ'), ('14 ST-UNION SQ'), ('34 ST-PENN STA')]
top_station_day_average_traffic = station_day_average_traffic[station_day_average_traffic['STATION'].isin(top_5_list)].sort_values('traffic', ascending = False)
print(top_station_day_average_traffic)
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
rc={'axes.labelsize': 16, 'font.size': 16, 'legend.fontsize': 32.0, 'axes.titlesize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
sns.set(rc = rc)
stagraph = sns.barplot(x = top_station_day_average_traffic['STATION'] , y = top_station_day_average_traffic['traffic'])
for item in stagraph.get_xticklabels():
    item.set_rotation(60)
ax.set_xlabel('Station')
ax.set_ylabel('Daily Traffic');
#ax.set_title('Traffic on Day of Week for NYC MTA System')
#fig.savefig("images/Daily Traffic for Top Stations.png")

def time_bin(x):
    if x < datetime.time(2):
        return "00:00-01:59"
    elif x < datetime.time(6):
        return "02:00-05:59"
    elif x < datetime.time(10):
        return "06:00-09:59"
    elif x < datetime.time(14):
        return "10:00-13:59"
    elif x < datetime.time(18):
        return "14:00-17:59"
    elif x < datetime.time(22):
        return "18:00- 21:59"
    else:
        return "22:00-23:59"
MTAdata["Time_Bin"] = MTAdata["time"].apply(time_bin)

time_breakdown = pd.DataFrame(MTAdata[MTAdata['STATION'].isin(top_5_list)].groupby(['STATION','Time_Bin']).sum()['traffic']).reset_index()

top_station_time_traffic = defaultdict(pd.DataFrame)
for station in top_5_list:
    top_station_time_traffic[station] = pd.DataFrame(MTAdata[MTAdata['STATION'] == station].groupby(['STATION', 'Time_Bin'])['traffic'].sum()).reset_index()
    top_station_time_traffic[station]['traffic'] = top_station_time_traffic[station]['traffic']/21
#    print(top_station_time_traffic[station].head())
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    graph = sns.barplot(x = top_station_time_traffic[station]['Time_Bin'], y = top_station_time_traffic[station]['traffic'])
    for item in graph.get_xticklabels():
        item.set_rotation(60)
    ax.set_xlabel('Time')
    ax.set_ylabel('Traffic')
    #fig.savefig("images/Peak hours for %s.png" %station)

top_station_turnstile_traffic = defaultdict(pd.DataFrame)

for station in top_5_list:
    top_station_turnstile_traffic[station] = pd.DataFrame(MTAdata[MTAdata['STATION'] == station].groupby(['turnstile'])['traffic'].sum()).reset_index()
    top_station_turnstile_traffic[station] = top_station_turnstile_traffic[station].sort_values('traffic', ascending = False).reset_index()
    top_station_turnstile_traffic[station]['traffic'] = top_station_turnstile_traffic[station]['traffic']/21
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    plt.tight_layout()
    graph = sns.barplot(y = top_station_turnstile_traffic[station]['turnstile'][:20], x = top_station_turnstile_traffic[station]['traffic'][:20])
    ax.set_xlabel('Traffic')
    ax.set_ylabel('Turnstiles')
    #fig.savefig("images/Highlight Turnstiles for %s.png" %station)

TechCompanyHeadcount = pd.read_csv('data/TechCompanyHeadcount.csv')
TechCompanyHeadcount.rename(columns=lambda x: x.strip(), inplace=True)
TechCompanyHeadcount.columns

fig, ax = plt.subplots(figsize = (10, 6))
techgraph = sns.barplot(x = TechCompanyHeadcount['Company Name'][:15], y = TechCompanyHeadcount['NYC headcount:'][:15], ax=ax)
for item in techgraph.get_xticklabels():
    item.set_rotation(60)
ax.set_xlabel('Company Name');
ax.set_ylabel('NYC Headcount');
#plt.savefig('images/NYC Tech Company Headcount.png');

Image(filename='images/Tech Company Map/Tech Company Map.png')

#Image('images/midtown.jpg')

gender_denisty_dict = {"Female": 51.7, "Male": 48.2}

age_density_dict = {"20-24":11.5, "25-29":15.9, "30-34":11.9, "35-39": 8.6}

genderdata = pd.DataFrame.from_dict(gender_denisty_dict, orient='index')
agedata = pd.DataFrame.from_dict(age_density_dict, orient='index')

genderdata['gender'] = ['Female','Male']
agedata['age'] = ['20-24', '25-29', '30-34', '35-39']

fig, ax = plt.subplots(figsize = (8, 6))
sns.barplot(y = genderdata[0], x = genderdata['gender'])
ax.set_xlabel('Gender');
ax.set_ylabel('Percentage(%)');
#plt.savefig('images/Gender Breakdown Midtown.png');

fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = agedata[0], x = agedata['age'])
ax.set_xlabel('Age');
ax.set_ylabel('Percentage(%)');
#plt.savefig('images/Age Breakdown Midtown.png');

midtown_demo = pd.read_csv('data/midtown_agg_demo.csv')

del midtown_demo['Unnamed: 3']
del midtown_demo['Unnamed: 4']
midtown_demo.dropna(inplace=True)

midtown_demo['Percent'] = midtown_demo['Percent'].apply(lambda x: str(x))
midtown_demo['Percent'] = midtown_demo['Percent'].apply(lambda x: float(x.strip("%")))
midtown_demo

fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = midtown_demo['Race'], x = midtown_demo['Percent'])
ax.set_ylabel('Race');
ax.set_xlabel('Percentage(%)');
#plt.savefig('images/Race Breakdown Midtown.png');

midtown_asian_pop = pd.read_csv('data/midtown_asian_breakdown.csv')

midtown_asian_pop.dropna(axis=1, inplace=True)
midtown_asian_pop.drop([0,14,15], inplace=True)

midtown_asian_pop = midtown_asian_pop.sort_values('Percent', ascending = False)

fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = midtown_asian_pop['SELECTED ASIAN SUBGROUPS'], x = midtown_asian_pop['Percent'])
ax.set_ylabel('Asian Sub Groups');
ax.set_xlabel('Percentage(%)');
#plt.savefig('images/Asian Sub Groups Breakdown Midtown.png');

midtown_hispanic_pop = pd.read_csv('data/midtown_hispanic_breakdown.csv')
midtown_hispanic_pop = midtown_hispanic_pop.sort_values('Percent', ascending = False)
midtown_hispanic_pop

midtown_hispanic_pop.dropna(axis=1, inplace=True)

fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = midtown_hispanic_pop['hispanic_subgroup'], x = midtown_hispanic_pop['Percent'])
ax.set_ylabel('Hispanic Sub Groups');
ax.set_xlabel('Percentage(%)');
#plt.savefig('images/Hispanic Sub Groups Breakdown Midtown.png');

midtown_income = pd.read_csv('data/midtown_income.csv')

midtown_income = midtown_income.iloc[::-1]
midtown_income['INCOME AND BENEFITS'] = ['$200,000 or more', '150,000  to 199,999','100,000 to 149,999', '75,000 to 99,999', '50,000 to 74,999', '35,000 to 49,999', '25,000 to 49,999', '15,000 to 24,999', '10,000 to 14,999', 'Less than$10,000']
midtown_income

fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = midtown_income['INCOME AND BENEFITS'], x = midtown_income['Percentage'])
ax.set_ylabel('Income');
ax.set_xlabel('Percentage(%)');
#plt.savefig('images/Income Breakdown Midtown.png');



