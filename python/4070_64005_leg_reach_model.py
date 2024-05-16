import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
cols = ['Name', 'Height', 'Reach', 'LegReach', 'Stance', 'Dob']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df = df.drop(['Name', 'Name_L'], axis=1)
df.head(3)

win_lose = df.Winner.append(df.Loser).drop_duplicates().reset_index()
win_lose.columns = ['index', 'Name']
win_lose

all3 = win_lose.merge(fighters, on='Name', how='left')[['Name', 'LegReach', 'Reach', 'Height']].dropna()
all3

y = all3.LegReach.values
X = all3[['Reach', 'Height']].values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(X, y)

lr.score(X, y)

from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X)
mean_squared_error(all3.LegReach, y_pred)

pd.DataFrame({'true':all3.LegReach, 'model':y_pred})

lr.coef_

lr.intercept_

def impute_legreach(r, h):
     return 0.16095475 * r + 0.42165158 * h - 0.901274878

pts = [(r, h, impute_legreach(r, h)) for r in np.linspace(60, 85) for h in np.linspace(60, 85)]
pts = pd.DataFrame(pts)
pts.columns = ['Reach', 'Height', 'LegReach']
pts

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all3.Reach, all3.Height, all3.LegReach)
ax.scatter(pts.Reach, pts.Height, pts.LegReach)
ax.set_xlabel('Reach')
ax.set_ylabel('Height')
ax.set_zlabel('LegReach')
ax.view_init(elev=0, azim=0)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all3.Reach, all3.Height, all3.LegReach)
ax.scatter(pts.Reach, pts.Height, pts.LegReach)
ax.set_xlabel('Reach')
ax.set_ylabel('Height')
ax.set_zlabel('LegReach')
ax.view_init(elev=0, azim=90)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all3.Reach, all3.Height, all3.LegReach)
ax.scatter(pts.Reach, pts.Height, pts.LegReach)
ax.set_xlabel('Reach')
ax.set_ylabel('Height')
ax.set_zlabel('LegReach')
ax.view_init(elev=16, azim=135)

import scipy
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(all3.Height.values, all3.LegReach.values)
slope, intercept, r_value, p_value, std_err

plt.plot(all3.Height.values, slope * all3.Height.values + intercept)
plt.scatter(all3.Height.values, all3.LegReach.values)
plt.xlabel('Height')
plt.ylabel('LegReach')

slope, intercept, r_value, p_value, std_err = linregress(x=all3.Height.values, y=all3.Reach.values)
slope, intercept, r_value, p_value, std_err

plt.plot(all3.Height.values, slope * all3.Height.values + intercept)
plt.scatter(all3.Height.values, all3.Reach.values)
plt.xlabel('Height')
plt.ylabel('Reach')

