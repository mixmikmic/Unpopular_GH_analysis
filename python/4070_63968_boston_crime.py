import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

df = pd.read_csv('/Users/jhalverson/Downloads/Crime_Incident_Reports.csv')
df.head().transpose()

df['FROMDATE'] = pd.to_datetime(df['FROMDATE'], infer_datetime_format=True)

df.dtypes

df.isnull().sum()

df.info()

incident_type = pd.crosstab(index=df["INCIDENT_TYPE_DESCRIPTION"], columns="count")
incident_type.columns = ["Count"]
incident_type.index = map(lambda x: x.title(), incident_type.index)
pd.options.display.max_rows = df["INCIDENT_TYPE_DESCRIPTION"].shape[0]
incident_type.sort_values('Count', ascending=False)

pd.reset_option('max_rows')

by_day = pd.crosstab(index=df["DAY_WEEK"], columns="count")
by_day.columns = ["Count"]
by_day

day_shoot = pd.crosstab(index=df["DAY_WEEK"], columns=df["Shooting"], margins=True)
day_shoot

print pd.options.display.max_rows
print pd.options.display.max_columns

pd.options.display.max_columns = 30
#pd.set_option('expand_frame_repr', True)

day_shoot_month = pd.crosstab(index=df["DAY_WEEK"], columns=[df["Shooting"], df["Month"]], margins=True)
day_shoot_month

pd.reset_option('max_columns')
#pd.reset_option('expand_frame_repr')

day_shoot = df[df['Shooting'] == 'Yes']
day_shoot_tab = pd.crosstab(index=day_shoot['FROMDATE'].apply(lambda x: x.hour), columns='Count')

plt.bar(day_shoot_tab.index, day_shoot_tab['Count'])
plt.xlabel('Hour of the day')
plt.ylabel('Number of shootings')
plt.xticks([0, 5, 10, 15, 20], ['12 am', '5 am', '10 am', '3 pm', '8 pm'])
plt.xlim(0, 24)

plt.fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 4))
shootings = []
hours = []
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i, day in enumerate(days):
    df_day = df[df.DAY_WEEK == day]
    day_shoot = pd.crosstab(index=df_day['FROMDATE'].apply(lambda x: x.hour), columns=df_day["Shooting"])
    shootings.extend(list(day_shoot.iloc[:,1].values))
    hours.extend(list(day_shoot.index + i * 24))
ax.bar(hours, shootings, width=1.0)
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Number of shootings')
ax.set_xlim(0, 168)
ax.set_ylim(0, 25)
for i, day in enumerate(days):
    plt.text(x=i * 24 + 12, y=20, s=day, transform=ax.transData, size=16, horizontalalignment='center')
for x in range(24, 168, 24):
    plt.axvline(x, ymin=0, ymax=25, linewidth=2, color='k', ls=':')
plt.xticks(range(0, 168 + 8, 8))
ax.set_xticklabels(7 * ['12 am', '8 am', '4 pm'] + ['12 am'])
plt.title('Total shootings in Boston from July 8, 2012 to August 10, 2015')
plt.tight_layout()
plt.savefig('shooting_by_day_by_hour.png')

#Statistical significance versus weekdays

df2014 = df[(df.FROMDATE > np.datetime64('2014-01-01')) & (df.FROMDATE < np.datetime64('2015-01-01'))]

df2014.iloc[0].FROMDATE

df2014.iloc[-1].FROMDATE

by_weapon = pd.crosstab(index=df[(df["WEAPONTYPE"] != 'None') &
                                 (df["WEAPONTYPE"] != 'Unarmed') &
                                 (df["WEAPONTYPE"] != 'Other')]["WEAPONTYPE"], columns="count")
by_weapon.columns = ["Count"]
by_weapon

labels = by_weapon.index
counts = by_weapon.Count
import matplotlib.colors as colors
clrs = colors.cnames.keys()[:counts.size]
explode = np.zeros(counts.size)

plt.pie(counts, explode=explode, labels=labels, colors=clrs, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')

x, y = zip(*df[df.Location != '(0.0, 0.0)']['Location'].map(eval).tolist())
plt.scatter(x, y, s=1, marker='.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

