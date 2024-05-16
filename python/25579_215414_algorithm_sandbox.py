import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

data = pd.read_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/final_wr.csv')
data.drop(['Unnamed: 0'], axis = 1, inplace = True)

data.head()

data[data.name == "Cecil Shorts"]

data[data.yac.isnull() == True]

data.isnull().sum()

print data.columns

test_df = data[data.season == 2015]

dyar_cols = ['DVOA', 'DYAR', 'name', 'first_down_ctchs', 'yac']
dyar_df = pd.DataFrame(test_df[dyar_cols])
dyar_df.sort('yac', ascending = False)

from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler

#data = df[df.season == 2015]

cols = ['name', 'rec_tds', 'rush_yds', 'rec_yards', 'DVOA', 'DYAR', 'yac', 'yards/reception',
       'ctch_pct', 'targets', 'drops', 'start_ratio', 'first_down_ctchs', 'recs_ovr_25',
       'receptions', 'y/tgt', 'EYds', 'dpis_drawn', 'dpi_yards', 'pct_team_tgts',
       'pct_team_receptions', 'pct_of_team_passyards', 'pct_team_touchdowns', 'fumbles']

scale_cols = ['rec_tds', 'rush_yds', 'rec_yards', 'DVOA', 'DYAR', 'yards/reception',
       'ctch_pct', 'targets', 'drops', 'start_ratio', 'first_down_ctchs',
       'receptions', 'y/tgt', 'EYds', 'dpis_drawn', 'dpi_yards', 'pct_team_tgts',
       'pct_team_receptions', 'pct_of_team_passyards', 'pct_team_touchdowns', 'yac']

sca = StandardScaler()
minmax = MinMaxScaler(feature_range = (1, 5), copy = False)

# for col in scale_cols:
#     data[col] = minmax.fit_transform(data[col])
#     #data[col] = data[col] + 1

### finding averages for the 2015 season

## average touchdowns per player
print np.average(data.rec_tds)

## average fumbles per player
print np.average(data.fumbles)

## average total yards (receiving plus rushing) per player
print np.average(data.rush_yds + data.rec_yards)

print np.average(data.rec_yards)

## average DVOA per player

print np.average(data.DVOA)

## average DYAR per player
print np.average(data.DYAR)

## average YAC per player
print np.average(data.yac)

## average yards per reception (yards/catch) per player
print np.average(data['yards/reception'])

## average catch rate per player
print np.average(data.ctch_pct)

## average targets per player
print np.average(data.targets)

## average number of drops per player
print np.average(data.drops)

## average start ratio per player
print np.average(data.start_ratio)

## average number of catches for first down per player
print np.average(data.first_down_ctchs)

## average receptions over 25 yards per player
print np.average(data.recs_ovr_25)

## average total receptions on the season per player
print np.average(data.receptions)

## average yards per target per player
print np.average(data['y/tgt'])

## average Expected yards per player
print np.average(data.EYds)

### This needs to be further examined as some of these values are way too big

## average defensive PI drawn per player
print np.average(data.dpis_drawn)

## average yards from DPI drawn per player
print np.average(data.dpi_yards)

## average percentage of team targets per player
print np.average(data.pct_team_tgts)

## average percentage of team receptions per player
print np.average(data.pct_team_receptions)

## average percentage of team passing yards per player
print np.average(data.pct_of_team_passyards)

## average percentage of team touchdowns per player
print np.average(data.pct_team_touchdowns)

amari_list = [['Amari Cooper', 2015, 21, '1-4', 'OAK', 210, 27.7, 16, 3, -3, -1.0, 0, -.2,
             130, 72, 1070, 14.86, 6, 66.9, .554, 8.23, 1, 0, 0, 0, 0, 1, 0, 'WR',
             5, 378, 45, .625, 68, 11, 10, 1029, -1.0, 1.22,
             'WR', 10.0, 31.50, 4.42, 33.0, 120.0, 3.98,
             6.71, 0, 34.0,3879.0,606.0,373.0,359.0, 21, 2015,
             73, 15/16, 3, 86, .21452145, .19303, .275844,
             .17647, 0]]
amari_df = pd.DataFrame(amari_list, columns = data.columns)
amari_df

data = data.append(amari_df, ignore_index = True)
data.tail()
    

data.columns

## PAR was our initial test metric to get a performance baseline
## It is pretty much every players performance in a category compared to the league
## average of that category

# data['PAR'] = (data.rec_tds/np.average(data.rec_tds) + 
#               (data.rush_yds + data.rec_yards)/np.average(data.rush_yds + data.rec_yards)+
#               data.DVOA/np.average(data.DVOA) +
#               data.DYAR/np.average(data.DYAR)+
#               data['yards/reception']/np.average(data['yards/reception'])+
#               data.ctch_pct/np.average(data.ctch_pct) -
#               data.drops/2+
#               data.start_ratio/np.average(data.start_ratio)+
#               data.first_down_ctchs/np.average(data.first_down_ctchs)+
#               data.recs_ovr_25/np.average(data.recs_ovr_25)+
#               data.receptions/np.average(data.receptions)+
#               data['y/tgt']/np.average(data['y/tgt'])+
#               data.dpis_drawn/np.average(data.dpis_drawn)+
#               data.dpi_yards/np.average(data.dpi_yards)+
#               data.pct_team_tgts/np.average(data.pct_team_tgts)+
#               data.pct_team_receptions/np.average(data.pct_team_receptions)+
#               data.pct_of_team_passyards/np.average(data.pct_of_team_passyards)+
#               data.pct_team_touchdowns/np.average(data.pct_team_touchdowns))


data['dropK'] = np.log(data['drops'] +1)
data['yacK'] = data.yac*(data.yac/data.rec_yards)
data['base'] = (((data.rec_yards+data.yacK+data.dpi_yards+(data.DYAR*100))*(data.receptions+(data.first_down_ctchs*data.first_down_ctchpct)+((data.recs_ovr_25**2)/data.receptions)))/(data.fumbles+data.dropK + (data.targets/data.pct_team_tgts))**2)
data['td_points'] = (((data.rec_tds+data.rush_tds)/np.average(data.rec_tds+data.rush_tds))*data.pct_team_touchdowns)
data['compilation'] = (data.base*100) + (data.td_points*7)

data.sort('compilation', ascending = False)







## We know the nulls are all coming from if a player has zero recieving yards so you 
## cannot divide by zero and you get a null value. So we are fine putting a zero here
data.yacK.fillna(value = 0, inplace = True)

## average score for all receivers in compilation score

print np.average(data.compilation[data.compilation >= 0])

data[data.name == "Amari Cooper"].sort('compilation', ascending = False)



plt.figure(figsize = (20, 20))
plt.xlabel('Distribution', fontsize = 30)
plt.ylabel('Number of points', fontsize = 30)
plt.hist(data.compilation[data.compilation >= 0], bins = 15)
plt.show()

import seaborn as sns

plt.figure(figsize = (20, 20))
sns.regplot(data.DVOA, data.DYAR)

plt.figure(figsize = (20, 20))
plt.hist(data.DVOA, bins = 15)
plt.show()

plt.figure(figsize = (20, 20))
sns.regplot(data.yac, data.yacK)

data.to_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/catcherr.csv')

len(data.describe().columns)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.grid_search import GridSearchCV
get_ipython().magic('matplotlib inline')

df = data[scale_cols]

train = df[df.yac.isnull() == False]
test = df[df.yac.isnull() == True]

# Pare down this list of features

X = train.drop('yac', axis = 1)
y = train.yac
kbest = SelectKBest(k=16)
kbest.fit(X,y)
# Show the feature importance for Kbest of 30
kbest_importance = pd.DataFrame(zip(X.columns, kbest.get_support()), columns = ['feature', 'important?'])

kbest_features = kbest_importance[kbest_importance['important?'] == True].feature
#Here's our dataframe
X_model = X[kbest_features]

x_train, x_test, y_train, y_test = train_test_split(X_model, y)
# Let the modelling begin

all_scores = {}
def evaluate_model(estimator, title):
    model = estimator.fit(x_train, y_train)
    print model.score(x_test, y_test)
    #y_pred = model.predict(x_test)
    #acc_score = accuracy_score(y_test, y_pred)
    #con_matrix = confusion_matrix(y_test, y_pred)
    #class_report = classification_report(y_test, y_pred)
    #print "Accuracy Score:", acc_score.round(8)
#     print
#     print "Confusion Matrix:\n", con_matrix
#     print
#     print "Classification Report:\n", class_report
    #all_scores[title] = acc_score
    #print all_scores


# Models to test
lr = LinearRegression()
dt = DecisionTreeRegressor()
xt = ExtraTreesRegressor()
knn = KNeighborsRegressor()
svr = SVR()
rfc = RandomForestRegressor()
ab = AdaBoostRegressor(base_estimator = dt)


evaluate_model(lr, 'LinearRegression')
evaluate_model(dt, 'Decision Tree')
evaluate_model(xt, 'Extra Trees')
evaluate_model(knn, 'KNeighbors')
evaluate_model(svr, 'SVR')
evaluate_model(rfc, 'Random Forest')
evaluate_model(ab, 'AdaBoost')

kbest_importance

lr_fit = lr.fit(X_model, y)
y_pred = lr_fit.predict(test[kbest_features])

test['y_pred'] = y_pred
test['yac'] = test.y_pred

test.head()

names_df = pd.DataFrame(data.name)

added = test.join(names_df)

added.sort('yac', ascending = False)



