# Standard Libraries Import
import math, os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the basemap package
from mpl_toolkits.basemap import Basemap
from IPython.display import set_matplotlib_formats
from mpl_toolkits import basemap

# Turn on retina display mode
set_matplotlib_formats('retina')
# turn off interactive mode
plt.ioff()

df = pd.read_excel("gtd_95to12_0617dist.xlsx", sheetname=0)

df_nepal = df[df['country_txt'] == 'Nepal']
df_nepal[['eventid', 'iyear', 'imonth', 'iday', 'country_txt', 'provstate', 'latitude',  'longitude', 'gname', 'nkill']].head()

# the total number of incidents occurred in Nepal

df_nepal.shape

# Since Communist Party of Nepal- Maoist (CPN-M) and Maoists are same groups, combining them

df_nepal['gname'] = df_nepal['gname'].replace('Communist Party of Nepal- Maoist (CPN-M)', 'Maoists')

# kathmandu coordinate : to keep the center of the map around Kathmandu
lon_0 = 85; lat_0 = 27
nepal = Basemap(projection='merc', area_thresh = 0.1, resolution='f',
                lat_0=lat_0,lon_0=lon_0,
                llcrnrlon=79,
                llcrnrlat=26.0,
                urcrnrlon=89,
                urcrnrlat=31.0)

fig = plt.figure(figsize=(18,12))

nepal.drawmapboundary(fill_color='aqua')

# Draw coastlines, and the edges of the map.
nepal.drawcoastlines(color='black')
nepal.fillcontinents(color='white', lake_color='aqua')
nepal.drawcountries(linewidth=1, linestyle='dashed' ,color='red')

# plotting the latitude, longitude data points in the map
xs, ys = list(df_nepal['longitude'].astype(float)), list(df_nepal['latitude'].astype(float))
x, y = nepal(xs, ys)
nepal.plot(x, y, 'bo')
plt.text(nepal(85, 29.5)[0], nepal(85, 29.5)[1], 'CHINA', fontsize=16, color='blue')
plt.text(nepal(84.7, 27.5)[0], nepal(84.7, 27.5)[1], 'NEPAL', fontsize=16, color='red')
plt.text(nepal(85, 26.3)[0], nepal(85, 26.3)[1], 'INDIA', fontsize=16, color='black')
plt.show()

# list of associated groups in Nepal

terror_groups = df_nepal['gname'].unique().tolist()
terror_groups[:10]

# creating a new DataFrame with group_name and total_events done by the group as two columns

my_dict = dict(df_nepal['gname'].value_counts())
group_events_df = pd.DataFrame(columns=['group_name', 'total_events'])

# adding the rows in the dataframe 
# Gropus causing less than 10 events will be merged into a new group: OTHERS
total_others_events = 0
for i, group in enumerate(terror_groups):
    total_events = my_dict[group]
    if total_events > 9:
        group_events_df.loc[i] = [group, total_events]
    else:
        total_others_events += total_events
        
# adding the new group: OTHERS
group_events_df.loc[i+1] = ['OTHERS', total_others_events]

# Now plotting the bar plot 
fig, ax = plt.subplots(figsize=(6, 12))
sns.barplot(x='total_events', y = 'group_name', data = group_events_df.sort_values('total_events', ascending=False))
ax.set(xlabel="Total Events", ylabel="")
plt.show()

# Finding the number of people killed by each of the group (nkill_group)
# and the number of people killed on their side (nkillter_group)

def num_killed_group(df = df_nepal):     # number of people killed by each group
    
    nkill_group_dict, nkillter_group_dict = dict(), dict()
    for group in terror_groups:
        nkill_group, nkillter_group = 0, 0
        for i in range(df.shape[0]):
            if df['gname'].tolist()[i] == group:
                if pd.isnull(df['nkill'].tolist()[i]): continue
                else: nkill_group += df['nkill'].tolist()[i]

            if df_nepal['gname'].tolist()[i] == group:
                if pd.isnull(df['nkillter'].tolist()[i]): continue
                else: nkillter_group += df['nkillter'].tolist()[i]

        nkill_group_dict[group] = nkill_group
        nkillter_group_dict[group] = nkillter_group
        
    return nkill_group_dict, nkillter_group_dict

nkill_group_dict, nkillter_group_dict = num_killed_group(df_nepal)

# Plotting the piechart of number of killings by each groups

n_killed, groups = [], []
for group in list(nkill_group_dict.keys()):
    nkilled = nkill_group_dict[group]
    if nkilled < 15: continue
    n_killed.append(nkilled)
    groups.append(group)
    
plt.figure(figsize=(12,12))
plt.pie(n_killed, labels=groups, autopct='%1.2f%%', shadow=True, startangle=150)
plt.axis('equal')
plt.title('GROUPWISE KILLING')
plt.show()

# Plotting the piechart of number of killings by each groups

nter_killed, groups = [], []
for group in list(nkillter_group_dict.keys()):
    nkilled = nkillter_group_dict[group]
    if nkilled < 15: continue
    nter_killed.append(nkilled)
    groups.append(group)
    
plt.figure(figsize=(12,12))
plt.pie(nter_killed, labels=groups, autopct='%1.2f%%', shadow=True, startangle=150)
plt.axis('equal')
plt.title('PERPETRATOR KILLING')
plt.show()

# creating a temporary DataFrame for the plotting data preparation

tmp_df = pd.DataFrame(columns = ['group', 'nkill', 'nkillter'])

# adding group column
tmp_df['group'] = nkill_group_dict.keys()

# adding nkill column
tmp_df['nkill'] = nkill_group_dict.values()

# adding nkillter columns
tmp_df['nkillter'] = nkillter_group_dict.values()

# splitting the long group name into different line so that
# they will be clearly seen while labelling in the plot along 
# the x-axis in the following plots

def group_name_rearrange():
    new_groups = []
    for group in tmp_df['group'].tolist():
        words = group.split()
        new_name = ''
        for i in range(len(words)):
            new_name += words[i] + ' '
            if i != 0 and i % 2 == 0: new_name += '\n'
        new_groups.append(new_name)
    return new_groups

# replacing the single line group naming by multiline group names
new_groups = group_name_rearrange()
tmp_df['group'] = new_groups

# sorting the data and taking only few to get clear graph
tmp_df = tmp_df.sort_values(['nkill', 'nkillter'], ascending=[False, False])[:7]

plt.figure(figsize=(18,6))
sns.barplot(x='group', y='nkill', data=tmp_df)
plt.title('Number of people killed by each gropus')
plt.show()

plt.figure(figsize=(15,6))
sns.barplot(x='group', y='nkillter', data=tmp_df[:5])
plt.title('Number of perpetrators killed from each group')
plt.show()

df_nepal[['gname', 'attacktype1_txt', 'weaptype1_txt', 'nkill']].head()

tmp=(pd.get_dummies(df_nepal[['gname', 'attacktype1_txt', 'nkill']].sort_values('nkill', ascending=False)).corr()[1:])
tmp = tmp[tmp.columns[56:]]
tmp.head()

plt.figure(figsize=(8,15))
ax = sns.heatmap(tmp, cmap='plasma', vmin=-0.1, vmax=1, annot=True)
plt.show()

df[['attacktype1_txt', 'weaptype1_txt']][df['gname'] == 'Maoists']['attacktype1_txt'].value_counts()

df[['attacktype1_txt', 'weaptype1_txt']][df['gname'] == 'Maoists']['weaptype1_txt'].value_counts()

tmp = df_nepal[['gname', 'attacktype1', 'attacktype1_txt', 'weaptype1_txt']]

# attacktype1 -> attacktype1_txt naming 

attack_type_dict = dict()
for attack in tmp['attacktype1_txt'].unique().tolist():
    code_value = tmp[tmp['attacktype1_txt'] == attack]['attacktype1'].unique().tolist()[0]
    attack_type_dict[attack] = code_value

attack_type_dict

f, ax = plt.subplots(figsize=(15, 12))
sns.despine(bottom=True, left=True)

sns.stripplot(x='attacktype1', y='gname', hue='weaptype1_txt', data=tmp)

ax.set(xlabel=False, ylabel=False)

plt.xlabel('Attacktype1')
plt.ylabel('Group Name')
plt.legend(loc='center')

plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(pd.get_dummies(df_nepal[['attacktype1_txt', 'weaptype1_txt']]).corr())
plt.show()

