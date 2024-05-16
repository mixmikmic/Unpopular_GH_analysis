import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/pivot_catcherr.csv')
data.head()

sns.regplot(data['rec_tds_2'], data['compilation_3'])
plt.show()

## creating a smaller dataframe with just the player name and the compilation 3 score
## to be used later to join with the PCA features
comp_df_cols = ['name', 'compilation_3']

comp_df = pd.DataFrame(data[comp_df_cols], columns = comp_df_cols)

comp_df.set_index(comp_df['name'], drop = True, inplace = True)
comp_df.drop('name', axis = 1, inplace = True)
comp_df

print data.columns

## I am dropping some of our engineered features since they will be co-linear with
## the other features that were used to create them

data.drop(['yacK_0', 'yacK_1', 'yacK_2', 'td_points_0', 'td_points_1', 'td_points_2', 'compilation_3'],
         axis = 1, inplace = True)
print data.columns

## the following loop will loop through every numeric column in the dataframe and 
## create a histogram of a random sample of that column and also perform a normal 
## test on that histogram

## the normal test sets the null hypothesis as "this sample comes from a normal distribution"
## so we would be able to accept that null hypothesis if the pvalue is greater than .05

## THIS IS BEING COMMENTED OUT BECAUSE IT TAKES A LONG TIME AND DOESNT NEED TO KEEP RUNNING


# import scipy
data_numeric = data[data.describe().columns]

# for i in data_numeric:
#     rand_sample = data_numeric[i].sample(100, random_state=6)
#     print i,':\n', scipy.stats.mstats.normaltest(rand_sample)
#     sns.distplot(data_numeric[i])
#     plt.xlabel(i)
#     plt.show()
#     print

## I will be performing PCA on all the numeric columns right now (data_numeric)

from sklearn import preprocessing

## standardizing all the columns

data_stand = preprocessing.StandardScaler().fit_transform(data_numeric)
data_stand

## creating the covariance matrix - this explains the variance between the different
## features within our dataframe

## for example, the value in the i,j position within the matrix explains the variance
## between the ith and the jth elements of a random vector, or between our features

cov_mat = np.cov(data_stand.T)

## creating my eigenvalues and corresponding eigenvectors

eigenValues, eigenVectors = np.linalg.eig(cov_mat)

print eigenValues 
print
print
print eigenVectors 

## creating the eigenpairs - just pairing the eigenvalue with its eigenvector
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]

## sort in ascending order and then reverse to descending (for clarification's sake)
# eigenPairs.sort()
# eigenPairs.reverse()

## loop through the eigenpairs and printing out the first row (eigenvalue)
## this is also seen in the code block above but just wanted to loop through again
## as it is a bit more clear like this
## I am also creating a list of the eigenvalues in ascending order to be able to reference it
sort_values = []
for i in eigenPairs:
    print i[0]
    sort_values.append(i[0])

## we have the eigenvalues above showing us feature correlation explanation, but it helps
## to see the cumulative variance explained as well, which i can show below

## need to sum the eigen values to get percentages
sumEigenvalues = sum(eigenValues)

## this is a percentage explanation
variance_explained = [(i/sumEigenvalues)*100 for i in sort_values]
variance_explained

### based on the above results, it seems that sticking to 46 features would be a decent
## cutoff point since the variance explained per feature drops below .3%

## this can very easily be manipulated by changing n_components adn then adding/subtracting
## columns to the dataframe in the code block below

## instantiate
pca = PCA(n_components = 46)

## fit and transform the standardized data
pca_cols = pca.fit_transform(data_stand)

## Here I am simply creating the column headers for the pca features
pca_col_list = []

for i in range(1, 47):
    pca_col_list.append('pca'+str(i))
    

## going to organize the columns into dataframe for organization
pca_df = pd.DataFrame(pca_cols, columns = pca_col_list)

##previewing dataframe
print pca_df.shape
pca_df.head()

## We used all of our columns to perform the PCA so we only need to join the names back on
## since we would not want to build a model off of the PCA features as well as the 
## original features that were used to construct the PCA columns

## I am going to set the index of our pca dataframe to the names of the related player

pca_df.set_index(data['name'], drop = False, inplace = True)
pca_df.head()

joined_df = pca_df.join(comp_df)
joined_df

bins = [-1, 10, 30, 60, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df.head()

## Now I will export this new dataframe as a CSV

joined_df.to_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/pca_catcherr.csv')

joined_df.shape

X = joined_df.drop(['compilation_3', 'categories'], axis = 1)
y = joined_df['categories']
print X.shape
print y.shape

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 11)
print X_train.shape
print y_train.shape
print
print X_test.shape
print y_test.shape

weighting = {'below average':1, 'league_average':3, 'quality starter':8, 'all_pro':1}

lr = LogisticRegression(penalty = 'l2', class_weight = weighting, C = 3, warm_start = True,
                       solver = 'lbfgs', multi_class = 'multinomial')

model = lr.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)
print model.score(X_test, y_test)
print classification_report(y_test, y_pred_lr)

rf = RandomForestClassifier(n_estimators = 8)

rf_model = rf.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print rf_model.score(X_test, y_test)
print classification_report(y_test, y_pred)

knn = KNeighborsClassifier(n_neighbors = 10, weights = 'distance')

knn_model = knn.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print knn_model.score(X_test, y_test)
print classification_report(y_test, y_pred_knn)

ada = AdaBoostClassifier(lr, n_estimators = 100, algorithm = 'SAMME.R')

ada_model = ada.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
print ada_model.score(X_test, y_test)
print classification_report(y_test, y_pred_lr)

from sklearn.svm import SVC

weighting = {'below average':.1, 'league_average':3, 'quality starter':10, 'all_pro':8}

svc = SVC(C = .7, class_weight = weighting, kernel = 'linear')

svc_model = svc.fit(X_train, y_train)
y_predsvc = svc_model.predict(X_test)
print svc_model.score(X_test, y_test)
print classification_report(y_test, y_predsvc)

## reading in both my standard dataframe and my dataframe of pca columns

df = pd.read_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/pivot_catcherr.csv')
pca_df = pd.read_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/pca_catcherr.csv')

# Create an average starts column
df['avg_starts'] = (df.start_ratio_0 + df.start_ratio_1 + df.start_ratio_2) / 3

#Create a column that adds up a player's dpi yards and penaltys drawn
df['dpis'] = df.dpis_drawn_0 + df.dpis_drawn_1 + df.dpis_drawn_2
df['dpi_yards'] = df.dpi_yards_0 + df.dpi_yards_1 + df.dpi_yards_2

df.head()

## this is a list of the features without any first year data


features_no_year_1 = ['age_2', 'weight_2', 'bmi_2',
             'rush_y/a_1', 'rush_y/a_2',
             'receptions_1', 'receptions_2',
            'rec_yards_1','rec_yards_2', 'rec_tds_1',
            'rec_tds_2', 'ctch_pct_1', 'ctch_pct_2',
             'first_down_ctchpct_1',
            'first_down_ctchpct_2',  'long_ctch_1', 'long_ctch_2',
             'drops_1', 'drops_2',  'EYds_1', 'EYds_2',
            'DVOA_1', 'DVOA_2', 'height_inches_2', 'avg_starts', 'dpis', 'dpi_yards',
             'pct_team_tgts_1',
            'pct_team_tgts_2', 'compilation_0', 'compilation_1', 'compilation_2', 'yacK_2',
                     'year_1_growth', 'year_2_growth']


# Create categories for player season_3 ratings

bins = [-1, 10, 30, 65, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
df['categories'] =  pd.cut(df['compilation_3'], bins, labels=labels)

## this shows us the average compilation score for players who had a score above zero
df[df.compilation_3 >0].compilation_3.mean()

from sklearn.preprocessing import scale

## going to create and scale a new data frame of just the feature columns we want to use
## for PCA

pca_df = df[features_no_year_1]
pca_df = scale(pca_df)

pca_df

## creating the covariance matrix - this explains the variance between the different
## features within our dataframe

## for example, the value in the i,j position within the matrix explains the variance
## between the ith and the jth elements of a random vector, or between our features

cov_mat = np.cov(pca_df.T)
cov_mat

## creating my eigenvalues and corresponding eigenvectors

eigenValues, eigenVectors = np.linalg.eig(cov_mat)

## creating the eigenpairs - just pairing the eigenvalue with its eigenvector
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]

## sort in ascending order and then reverse to descending (for clarification's sake)
# eigenPairs.sort()
# eigenPairs.reverse()

## loop through the eigenpairs and printing out the first row (eigenvalue)
## this is also seen in the code block above but just wanted to loop through again
## as it is a bit more clear like this
## I am also creating a list of the eigenvalues in ascending order to be able to reference it
sort_values = []
for i in eigenPairs:
    print i[0]
    sort_values.append(i[0])

## we have the eigenvalues above showing us feature correlation explanation, but it helps
## to see the cumulative variance explained as well, which i can show below

## need to sum the eigen values to get percentages
sumEigenvalues = sum(eigenValues)

## this is a percentage explanation
variance_explained = [(i/sumEigenvalues)*100 for i in sort_values]
variance_explained

### based on the above results, it seems that sticking to 16 features would be a decent
## cutoff point since the variance explained per feature drops below 1%

## this can very easily be manipulated by changing n_components adn then adding/subtracting
## columns to the dataframe in the code block below

## instantiate
pca = PCA(n_components = 16)

## fit and transform the standardized data
pca_cols = pca.fit_transform(pca_df)

## Here I am simply creating the column headers for the pca features
pca_col_list = []

for i in range(1, 17):
    pca_col_list.append('pca'+str(i))

## going to organize the columns into dataframe for organization
pca_df = pd.DataFrame(pca_cols, columns = pca_col_list)

##previewing dataframe
print pca_df.shape
pca_df.head()

## We used all of our columns to perform the PCA so we only need to join the names back on
## since we would not want to build a model off of the PCA features as well as the 
## original features that were used to construct the PCA columns

## I am going to set the index of our pca dataframe to the names of the related player

pca_df.set_index(df['name'], drop = False, inplace = True)
pca_df.head()

joined_df = pca_df.join(comp_df)
joined_df

# Create categories for player season_3 ratings

bins = [-1, 10, 30, 65, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df

## i am splitting the compilation scores into bins to separate recievers into 'talent pools'

bins = [-1, 10, 30, 60, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df.head()

## setting my X and y in order to build a model off of the data

X = joined_df.drop(['compilation_3', 'categories'], axis = 1)
y = joined_df['categories']
print X.shape
print y.shape

# splitting data into train and test section
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state = 12)

## I am going to weigth each category to try to more accurately predict that bin
cat_weights = {'below average':1, 'league_average':8, 'quality starter':5, 'all_pro':.5}

svc = SVC(C = .8, kernel = 'linear', shrinking = True, class_weight = cat_weights)

from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
cvp = cross_val_predict(svc, X, y, cv = 10)

print classification_report(y, cvp)

## support vector machine classifier

svc = SVC(C = .3, class_weight=cat_weights, probability = True, kernel='poly', degree = 2)
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
preds = svc.predict(X_test)
print classification_report(y_test, preds)

cat_weights = {'below average':.4, 'league_average':3, 'quality starter':5, 'all_pro':4}

lr = LogisticRegression(C=1, solver = 'lbfgs', multi_class = 'multinomial', penalty='l2', class_weight = cat_weights, random_state=11)
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
preds = lr.predict(X_test)
print classification_report(y_test, preds)

cat_weights = {'below average':.4, 'league_average':5, 'quality starter':5, 'all_pro':2}


ab = AdaBoostClassifier(base_estimator = lr, n_estimators = 25, random_state=11)
ab.fit(X_train, y_train)
ab.score(X_test, y_test)
preds = ab.predict(X_test)
print classification_report(y_test, preds)
labels = svc.predict(X)

## creating dataframe to perform LDA on
lda_df = df[features_no_year_1]
lda_df.head()

lda_df.set_index(df['name'], drop = False, inplace = True)
lda_df.head()

joined_df = lda_df.join(comp_df)
joined_df.head()

# Create categories for player season_3 ratings

bins = [-1, 10, 30, 65, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df

from sklearn.lda import LDA

lda = LDA(n_components=4)

X = scale(joined_df.drop(['compilation_3', 'categories'], axis = 1))
y = joined_df['categories']

## fit and transform the standardized data
lda_cols = lda.fit_transform(X, y)

lda_cols.shape

lda_df = pd.DataFrame(lda_cols, columns = ['lda1', 'lda2', 'lda3'])

##previewing dataframe
print lda_df.shape
lda_df.head()

lda_df.set_index(df['name'], drop = False, inplace = True)
lda_df.head()

joined_df = lda_df.join(comp_df)
joined_df.head()

bins = [-1, 10, 30, 65, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df.head()

## setting my X and y in order to build a model off of the data

X = joined_df.drop(['compilation_3', 'categories'], axis = 1)
y = joined_df['categories']
print X.shape
print y.shape

# splitting data into train and test section
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .35)

## I am going to weigth each category to try to more accurately predict that bin
cat_weights = {'below average':1, 'league_average':8, 'quality starter':4, 'all_pro':5}

## support vector machine classifier

svc = SVC(C = .8, class_weight=cat_weights, probability = True, kernel='linear', degree = 1, shrinking = True)
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
preds = svc.predict(X_test)
print classification_report(y_test, preds)
print recall_score(y_test, preds, average = 'macro')

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score

c_list = [x/10.0 for x in range(1, 31, 1)]

parameters = {'C':c_list, 'kernel':['rbf', 'linear', 'poly', 'sigmoid'], 'degree':range(1,7)}

svc = SVC(class_weight=cat_weights, probability = True)
gsv_svc = GridSearchCV(svc, param_grid = parameters, scoring = 'recall_macro', n_jobs = -1,
                      cv = 3, verbose = 1)

gsv_fit = gsv_svc.fit(X, y)

gsv_fit.best_estimator_

gsv_fit.best_score_



