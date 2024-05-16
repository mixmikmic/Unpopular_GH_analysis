import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().magic('matplotlib inline')

df = pd.read_csv('cleaned_wrs.csv')

df.columns
df.isnull().sum()

sns.stripplot(x='years_in_league', y='receptions', data=df[df.years_in_league <=3], jitter=0.3)

sns.stripplot(x='years_in_league', y='y/tgt', data =df[df.years_in_league <=3], jitter=0.3)

for i in range(0,4):
    plt.hist(df[df.years_in_league == i]['y/tgt'], color='green', alpha=0.6)
    plt.hist(df[df.years_in_league == i]['yards/reception'], color = 'goldenrod', alpha = 0.6)
    plt.legend()
    plt.show()

df.columns

for i in range(0,4):
    plt.hist(df[df.years_in_league == i]['ctch_pct'], color='green', alpha=0.6)
    plt.hist(df[df.years_in_league == i]['first_down_ctchpct'], color = 'goldenrod', alpha = 0.6)
    plt.legend()
    plt.show()
    

for i in range(0,4):
    sns.distplot(df[df.years_in_league ==i].drops)
    plt.show()

df_15 = df[df.season == 2015]

df_15.reset_index(inplace=True)
df_15.head()

plt.scatter(df_15.ctch_pct, df_15.drops)

from sklearn.cluster import KMeans
df.columns

kmeans = KMeans(n_clusters = 5)

features = ['weight', 'bmi', 'rush_atts', 'rush_yds', 'rush_tds', 'targets', 'receptions', 'rec_yards', 'rec_tds', 'fumbles', 
           'pro_bowls', 'all_pros', 'yac', 'first_down_ctchs', 'recs_ovr_25', 'drops', 'height_inches', 'start_ratio',
           'dpis_drawn', 'dpi_yards']
X = df_15[features]
kmeans.fit_predict(X)

kmeans.labels_

kmeans.labels_.T

label_df = pd.DataFrame(kmeans.labels_.T, columns = ['player_label'])

df2 = df_15.join(label_df)

df2.tail(25)

# First, create a dataframe where the wr advanced analytic columns are all filled. 

advanced_df = df[(df.EYds.isnull() == False)]

advanced_df.head()

advanced_df = advanced_df[advanced_df.pct_team_tgts.isnull() == False]

advanced_df.reset_index(inplace=True, drop=True)

advanced_df.isnull().sum()

advanced_df.columns

features = ['targets', 'receptions', 'rec_tds', 'start_ratio', 'pct_team_tgts', 'pct_team_receptions', 'pct_team_touchdowns',
            'rec_yards', 'dpi_yards', 'fumbles', 'years_in_league', 'recs_ovr_25', 'first_down_ctchs', 'pct_of_team_passyards']

from sklearn.preprocessing import scale
X = scale(advanced_df[features])
y = advanced_df.DVOA

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

x_train, x_test, y_train, y_test = train_test_split(X,y)

def evaluate_model(estimator, title):
    model = estimator.fit(x_train, y_train)
    score = estimator.score(x_test, y_test)
    print 'The %r model scored %.4f.' % (title, score)

lr = LinearRegression()
rfr = RandomForestRegressor()
svr = SVR()
knr = KNeighborsRegressor()
br = BayesianRidge()

br.fit(x_train, y_train)
plt.scatter(br.predict(x_test), br.predict(x_test)-y_test)
plt.title('Linear Regression with a score of %.4f' % br.score(x_test, y_test))

lr.fit(x_train, y_train)
plt.scatter(lr.predict(x_test), lr.predict(x_test)-y_test)
plt.title('Linear Regression with a score of %.4f' % lr.score(x_test, y_test))

svr.fit(x_train, y_train)
plt.scatter(svr.predict(x_test), svr.predict(x_test)-y_test)
plt.title('SVR with a score of %.4f' % svr.score(x_test,y_test))

knr.fit(x_train, y_train)
plt.scatter(knr.predict(x_test), knr.predict(x_test)-y_test)
plt.title('KNR with a score of %.4f' % knr.score(x_test, y_test))

rfr.fit(x_train, y_train)
plt.scatter(rfr.predict(x_test), rfr.predict(x_test)-y_test)
plt.title('Random forest with a score of %.4f'% rfr.score(x_test, y_test))

svr2 = SVR(C=4, epsilon=0.04)

svr2.fit(x_train, y_train)
plt.scatter(svr2.predict(x_test), svr2.predict(x_test)-y_test)
plt.title('SVR residuals with a score of %.4f' % svr2.score(x_test,y_test))

X2 = scale(advanced_df[features])
y2 = advanced_df.DYAR
x_train, x_test, y_train, y_test = train_test_split(X2,y2)
from scipy import stats
import scipy
def r2(x,y):
    return stats.pearsonr(x,y)[0]**2

br.fit(x_train, y_train)
sns.regplot(br.predict(x_test), br.predict(x_test)-y_test, fit_reg = False)
plt.title('Bayesian Ridge Regression with a score of %.4f' % br.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(br.predict(x_test), br.predict(x_test)-y_test)
print r_value**2

lr.fit(x_train, y_train)
plt.scatter(lr.predict(x_test), lr.predict(x_test)-y_test)
plt.title('Linear Regression with a score of %.4f' % lr.score(x_test, y_test))

svr.fit(x_train, y_train)
plt.scatter(svr.predict(x_test), svr.predict(x_test)-y_test)
plt.title('SVR with a score of %.4f' % svr.score(x_test,y_test))

knr.fit(x_train, y_train)
plt.scatter(knr.predict(x_test), knr.predict(x_test)-y_test)
plt.title('KNR with a score of %.4f' % knr.score(x_test, y_test))

rfr.fit(x_train, y_train)
plt.scatter(rfr.predict(x_test), rfr.predict(x_test)-y_test)
plt.title('Random forest with a score of %.4f'% rfr.score(x_test, y_test))

X2 = scale(advanced_df[features])
y2 = advanced_df.EYds
x_train, x_test, y_train, y_test = train_test_split(X2,y2)

br.fit(x_train, y_train)
sns.regplot(br.predict(x_test), br.predict(x_test)-y_test, fit_reg = False)
plt.title('Bayesian Ridge Regression with a score of %.5f' % br.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(br.predict(x_test), br.predict(x_test)-y_test)
print r_value**2

lr.fit(x_train, y_train)
plt.scatter(lr.predict(x_test), lr.predict(x_test)-y_test)
plt.title('Linear Regression with a score of %.5f' % lr.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lr.predict(x_test), lr.predict(x_test)-y_test)
print r_value**2

svr.fit(x_train, y_train)
plt.scatter(svr.predict(x_test), svr.predict(x_test)-y_test)
plt.title('SVR with a score of %.4f' % svr.score(x_test,y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(svr.predict(x_test), svr.predict(x_test)-y_test)
print r_value**2

knr.fit(x_train, y_train)
plt.scatter(knr.predict(x_test), knr.predict(x_test)-y_test)
plt.title('KNR with a score of %.4f' % knr.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(knr.predict(x_test), knr.predict(x_test)-y_test)
print r_value**2

rfr.fit(x_train, y_train)
plt.scatter(rfr.predict(x_test), rfr.predict(x_test)-y_test)
plt.title('Random forest with a score of %.4f'% rfr.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(rfr.predict(x_test), rfr.predict(x_test)-y_test)
print r_value**2

df = pd.read_csv('cleaned_wrs.csv')

### Imputing DVOA

train = df[(df.DVOA.isnull() ==False) & (df.pct_team_tgts.isnull() == False)]
train.reset_index(inplace=True, drop=True)
test = df[(df.DVOA.isnull() == True) & (df.pct_team_tgts.isnull() == False)]
test.reset_index(inplace= True, drop=True)

features = ['targets', 'receptions', 'rec_tds', 'start_ratio', 'pct_team_tgts', 'pct_team_receptions', 'pct_team_touchdowns',
            'rec_yards', 'dpi_yards', 'fumbles', 'years_in_league', 'recs_ovr_25', 'first_down_ctchs', 'pct_of_team_passyards']
X = train[features]
y = train.DVOA

# Our best model for predicting DVOA was a support vector regressor. We'll fit this model on the 
svr = SVR(C=4, epsilon=0.04)
svr.fit(X,y)
dvoa_predictions = pd.DataFrame(svr.predict(test[features]), columns=['DVOA_predicts'])

test = test.join(dvoa_predictions)

test['DVOA'] = test['DVOA_predicts']

test.drop('DVOA_predicts', inplace=True, axis=1)

frames = [train, test]
df = pd.concat(frames, axis=0, ignore_index=True)

### Imputing DYAR

train = df[(df.DYAR.isnull() ==False) & (df.pct_team_tgts.isnull() == False)]
train.reset_index(inplace=True, drop=True)
test = df[(df.DYAR.isnull() == True) & (df.pct_team_tgts.isnull() == False)]
test.reset_index(inplace= True, drop=True)

features = ['targets', 'receptions', 'rec_tds', 'start_ratio', 'pct_team_tgts', 'pct_team_receptions', 'pct_team_touchdowns',
            'rec_yards', 'dpi_yards', 'fumbles', 'years_in_league', 'recs_ovr_25', 'first_down_ctchs', 'pct_of_team_passyards']
X = train[features]
y = train.DYAR

# Our best model for predicting DYAR was a Bayesian Ridge Regressor 
br = BayesianRidge()
br.fit(X,y)
dyar_predictions = pd.DataFrame(br.predict(test[features]), columns = ['DYAR_predicts'])

test = test.join(dyar_predictions)
test['DYAR'] = test['DYAR_predicts']
test.drop('DYAR_predicts', inplace=True, axis=1)

frames = [train,test]
df = pd.concat(frames, axis=0, ignore_index=True)
df.head()

### Imputing EYds

train = df[(df.EYds.isnull() ==False) & (df.pct_team_tgts.isnull() == False)]
train.reset_index(inplace=True, drop=True)
test = df[(df.EYds.isnull() == True) & (df.pct_team_tgts.isnull() == False)]
test.reset_index(inplace= True, drop=True)

# A Bayesian Ridge was also our best predictor for EYds. In general, we're able to most confidently predict EYds.
X = train[features]
y = train.EYds

br.fit(X,y)
eyds_predictions = pd.DataFrame(br.predict(test[features]), columns = ['EYds_predicts'])

test = test.join(eyds_predictions)
test['EYds'] = test['EYds_predicts']
test.drop('EYds_predicts', inplace=True, axis=1)

frames = [train, test]
df = pd.concat(frames, axis=0, ignore_index=True)
df.isnull().sum()

df.to_csv('wrs_finalish.csv')



