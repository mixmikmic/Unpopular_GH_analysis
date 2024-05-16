get_ipython().magic('matplotlib inline')

# Imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
#import seaborn as sns
import sklearn
import numpy as np

# Import data
co2_df = pd.read_csv('input/global_co2.csv')
temp_df = pd.read_csv('input/annual_temp.csv')
print(co2_df.head())
print(temp_df.head())
#print(co2_df.columns=['Year','CO2'])

# Clean data
co2_df = co2_df.ix[:,:2]                     # Keep only total CO2
co2_df = co2_df.ix[co2_df['Year'] >= 1960]   # Keep only 1960 - 2010
co2_df.columns=['Year','CO2']                # Rename columns
co2_df = co2_df.reset_index(drop=True)                # Reset index
print(co2_df.tail())

temp_df =temp_df[temp_df.Source != 'GISTEMP'] # Keep only one source 
temp_df.drop('Source', inplace = True, axis = 1) #Drop name of source
temp_df = temp_df.reindex(index=temp_df.index[::-1]) #Reset Index

temp_df = temp_df.ix[temp_df['Year'] >= 1960]   # Keep only 1960 - 2010
temp_df.columns=['Year', 'Temperature']# Rename columns
temp_df = temp_df.reset_index(drop=True)     
print(temp_df.tail())

# Concatenate
climate_change_df = pd.concat([co2_df, temp_df.Temperature], axis=1)
print(climate_change_df.head())

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
fig.set_size_inches(12.5,7.5)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(climate_change_df['Year'], climate_change_df['Temperature'] , climate_change_df['CO2'])
ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
ax.view_init(10, -45)

f, axarr = plt.subplots(2, sharex=True)
f.set_size_inches(12.5,7.5)
#help(plt.subplots)
axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
axarr[1].set_xlabel('Year')
axarr[1].set_ylabel('Relative temperature')

from sklearn.model_selection import train_test_split

climate_change_df = climate_change_df.dropna()

X = climate_change_df.as_matrix(['Year'])
Y = climate_change_df.as_matrix(['CO2', 'Temperature']).astype('float32')

indexes = ~np.isnan(X)
X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(X,Y, test_size=0.1)) 
#print(X_train)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, Y_train)
print('Score: ', reg.score(X_test.reshape(-1, 1), Y_test))

x_line = np.arange(1960,2011).reshape(-1,1)
p = reg.predict(x_line).T
print(x_line)


fig2 = plt.figure()
fig2.set_size_inches(12.5, 7.5)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(xs=climate_change_df['Year'], ys=climate_change_df['Temperature'], zs=climate_change_df['CO2'])
ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
ax.plot(x_line, p[1], p[0], color='g')
ax.view_init(10, -45)

f, axarr = plt.subplots(2, sharex=True)
f.set_size_inches(12.5, 7.5)
axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
axarr[0].plot(x_line, p[0])
axarr[0].set_ylabel('CO2 Emissions')
axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
axarr[1].plot(x_line, p[1])
axarr[1].set_xlabel('Year')
axarr[1].set_ylabel('Relative temperature')

