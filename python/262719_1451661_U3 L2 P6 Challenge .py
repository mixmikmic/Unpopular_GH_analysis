import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import ensemble
from sklearn import tree
import pydotplus
import graphviz
import time
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('seattleWeather_1948-2017.csv')

print(df.shape)
df.head()

df.dtypes

# Change Rain to binary in order to build model.
df['RAIN'] = df['RAIN'].map(lambda i: 1 if i == True else 0)

# Had missing values before so going to drop some valeus 
df = df.dropna()

df.head(80)

# decision tree
# Initialize and train our tree using PCA.
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
X = df.drop(['RAIN', 'DATE'], axis = 1)
Y = df['RAIN']

# Use PCA to create new columns 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
pca_X = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

dt_start_time = time.time()
decision_tree = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_features=1,
    max_depth=4,
)

decision_tree.fit(pca_X, Y)
display(cross_val_score(decision_tree, pca_X, Y, cv = 10))
print ("Decision Tree runtime: {}".format(time.time() - dt_start_time))

# random forest using PCA components
rf_start_time = time.time()
rfc = ensemble.RandomForestClassifier()
display(cross_val_score(rfc, pca_X, Y, cv=10))
print ("Random Forest runtime: {}".format(time.time() - rf_start_time))

# write up analysis as most cases for parameters for random forest, and the decision tree.
# Decision Tree 2 with SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# feature extraction 
test = SelectKBest(k=3) 
X1 = X
fit = test.fit(X1, Y)

# Identify features with highest score from a predictive perspective (for all programs) 
names2 = X1.columns
best_features = pd.DataFrame(fit.scores_, index = names2) 
best_features.columns = ['Best Features'] 
best_features.sort_values(by=['Best Features'], ascending=False)

# Take a look at what the best features are.
print(best_features)

# Make a new dataframe with only PRCP and TMAX since they are the stongest features.
selected_features = df[['PRCP', 'TMAX']]

dt_start_time = time.time()
decision_tree = tree.DecisionTreeClassifier()
    
display(cross_val_score(decision_tree, selected_features, Y, cv = 10))
print ("Decision Tree runtime: {}".format(time.time() - dt_start_time))

# Random forest after using SelectKBest features.
rf_start_time = time.time()

display(cross_val_score(rfc, selected_features, Y, cv=10))
print ("Random Forest runtime: {}".format(time.time() - rf_start_time))

# Grid Search CV for decision tree
from sklearn.grid_search import GridSearchCV
# Set parameter grid range.
param_grid = {'max_depth':[10,25,50,75,100,125,150,175,200,300,400,500]}

# Set up the decision tree for X and Y.
decision_tree = tree.DecisionTreeClassifier()

# Grid Search for decision tree
grid_DT = GridSearchCV(decision_tree, param_grid, cv=10, verbose=3)

grid_DT.fit(X, Y)

# summarize the results of the grid search
# View the accuracy score
print('Best score for data:', grid_DT.best_score_) 

#GridSearchCV for random forest 
param_grid = {'n_estimators':[10,25,50,75,100,125,150,175,200,300,400,500]}

# Prepare the random forest
rfc = ensemble.RandomForestClassifier()

# Start the grid search again
grid_DT = GridSearchCV(rfc, param_grid, cv=10, verbose=3)

grid_DT.fit(X, Y)

# summarize the results of the grid search
# View the accuracy score
print('Best score for data:', grid_DT.best_score_) 



