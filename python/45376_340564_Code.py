import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import itertools
import pickle
# import openpyxl as px
# from pyexcel_xls import get_data

get_ipython().magic('matplotlib inline')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
CSVdata=pd.read_csv('Building Electrical.csv', parse_dates=[0], date_parser=dateparse)

data=pd.read_csv('Building Electrical.csv', parse_dates=[0], date_parser=dateparse)
data['Hour']=data['Timestamp'].dt.hour
data['Date']=data['Timestamp'].dt.date
data['Date1']=data['Timestamp'].dt.date
data['Porter Hall Electric Real Power']=data['Porter Hall Electric Real Power'].convert_objects(convert_numeric=True)
data

CSVdata.set_index('Timestamp', drop=True, append=False, inplace=True, verify_integrity=False)
CSVdata

CSVdata.drop('Baker Hall Electric Real Power',axis=1, inplace=True)
CSVdata['Porter Hall Electric Real Power'] = CSVdata['Porter Hall Electric Real Power'].convert_objects(convert_numeric=True)

resampled_data=CSVdata.resample('5T').mean()
resampled_data

filled_data=resampled_data.interpolate()
filled_data.isnull().sum().sum()

fig1=plt.figure(figsize=(10,5))
plt.plot(filled_data['Porter Hall Electric Real Power'])
plt.title('Porter Hall daily electricity consumption')
plt.show()

fig2=plt.figure(figsize=(10,5))
plt.title('Hunt Library daily electricity consumption')
plt.plot(filled_data['Hunt Library Real Power'])
plt.show()


data_groupbyHour=data.groupby(['Hour']).mean()
data_groupbyHour

plt.plot(data_groupbyHour['Porter Hall Electric Real Power'])
plt.title('Porter Hall hourly consumption')
plt.xlabel('Hour')
plt.ylabel('Porter Hall Electric Real Power')
plt.show()

plt.plot(data_groupbyHour['Hunt Library Real Power'])
plt.title('Hunt Library hourly consumption')
plt.xlabel('Hour')
plt.ylabel('Hunt Library Real Powe')
plt.show()

fig6=plt.figure()
ax1=plt.subplot()
ax1.plot(data_groupbyHour['Porter Hall Electric Real Power'],color='b')
plt.ylabel('Porter Hall Electric Real Power')
plt.xlabel('Hour')

ax2=ax1.twinx()
ax2.plot(data_groupbyHour['Hunt Library Real Power'],color='r')
plt.ylabel('Hunt Library Real Power')
plt.xlabel('Hour')
plt.legend()
plt.show()

data_groupbyDate=data.groupby(['Date']).mean()
data_groupbyDate

fig3=plt.figure(figsize=(12,5))
plt.plot(data_groupbyDate['Porter Hall Electric Real Power'])
plt.title('Porter Hall daily consumption')
plt.ylabel('Porter Hall Electric Real Power')
plt.show()

fig4=plt.figure(figsize=(12,5))
plt.title('Hunt Library daily consumption')
plt.plot(data_groupbyDate['Hunt Library Real Power'])
plt.ylabel('Hunt Library Electric Real Power')
plt.show()

fig5=plt.figure(figsize=(12,5))
ax1=plt.subplot()
ax1.plot(data_groupbyDate['Porter Hall Electric Real Power'],color='b')
plt.ylabel('Porter Hall daily consumption')
plt.xlabel('Date')

ax2=ax1.twinx()
ax2.plot(data_groupbyDate['Hunt Library Real Power'],color='r')
plt.ylabel('Hunt Library daily consumption')
plt.xlabel('Date')
plt.legend()
plt.show()

data['DayOfYear'] = data['Timestamp'].dt.dayofyear
loadCurves1 = data.groupby(['DayOfYear', 'Hour'])['Porter Hall Electric Real Power'].mean().unstack()
loadCurves2 = data.groupby(['DayOfYear', 'Hour'])['Hunt Library Real Power'].mean().unstack()

import matplotlib.colors as clrs
plt.imshow(loadCurves1, aspect='auto',cmap='summer')
plt.title('Heatmap of Porter Hall Electric Consumption')
plt.ylabel('Day of Year')
plt.xlabel('Hour of the Day')
plt.colorbar()

plt.imshow(loadCurves2, aspect='auto',cmap='summer')
plt.title('Heatmap of Hunt Library Electric Consumption')
plt.ylabel('Day of Year')
plt.xlabel('Hour of the Day')
plt.colorbar()

data_groupbyDate

def plot_regdataOfPorter():
    plt.plot(data_groupbyDate['DayOfYear'],data_groupbyDate['Porter Hall Electric Real Power'],'rd')
    plt.xlabel('DayOfYear')
    plt.ylabel('Porter Hall Electric Real Power')
def plot_regdataOfHunt():
    plt.plot(data_groupbyDate['DayOfYear'],data_groupbyDate['Hunt Library Real Power'],'rd')
    plt.xlabel('DayOfYear')
    plt.ylabel('Hunt Library Real Power')

from sklearn import tree
x = data_groupbyDate['DayOfYear']
y = data_groupbyDate['Porter Hall Electric Real Power']
xrange = np.arange(x.min(),x.max(),(x.max()-x.min())/100).reshape(100,1)
x = x[:, None]
reg = tree.DecisionTreeRegressor() # Default parameters, though you can tweak these!
reg.fit(x,y)

plot_regdataOfPorter()
plt.title('Regression of Porter Hall Electric Consumption')
plt.plot(xrange,reg.predict(xrange),'b--',linewidth=3)
plt.show()

print(reg.score(x,y))

x = data_groupbyDate['DayOfYear']
y = data_groupbyDate['Hunt Library Real Power']
xrange = np.arange(x.min(),x.max(),(x.max()-x.min())/100).reshape(100,1)
x = x[:, None]
reg = tree.DecisionTreeRegressor() # Default parameters, though you can tweak these!
reg.fit(x,y)

plot_regdataOfHunt()
plt.title('Regression of Hunt Library Consumption')
plt.plot(xrange,reg.predict(xrange),'b--',linewidth=3)
plt.show()

print(reg.score(x,y))





