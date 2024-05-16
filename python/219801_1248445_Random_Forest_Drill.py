import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Replace the path with the correct path for your data.
y2015 = pd.read_csv(
    'https://www.dropbox.com/s/0so14yudedjmm5m/LoanStats3d.csv?dl=1',
    skipinitialspace=True,
    header=1
)

# Note the warning about dtypes.

y2015.head()

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = y2015.drop('loan_status', 1)
Y = y2015['loan_status']
# X = pd.get_dummies(X)

# cross_val_score(rfc, X, Y, cv=5)

categorical = y2015.select_dtypes(include=['object'])
for i in categorical:
    column = categorical[i]
    print(i)
    print(column.nunique())

# Convert ID and Interest Rate to numeric.
y2015['id'] = pd.to_numeric(y2015['id'], errors='coerce')
y2015['int_rate'] = pd.to_numeric(y2015['int_rate'].str.strip('%'), errors='coerce')

# Drop other columns with many unique variables
y2015.drop(['url', 'emp_title', 'zip_code', 'earliest_cr_line', 'revol_util',
            'sub_grade', 'addr_state', 'desc'], 1, inplace=True)

y2015.tail()

# Remove two summary rows at the end that don't actually contain data.
y2015 = y2015[:-2]

pd.options.display.max_columns = 300
pd.get_dummies(y2015).head(5)

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = y2015.drop('loan_status', 1)
Y = y2015['loan_status']
X = pd.get_dummies(X)
X = X.dropna(axis=1)

cross_val_score(rfc, X, Y, cv=10)

#Correlation Matrix

correlation_matrix = X.corr()
display(correlation_matrix)

correlation_matrix_filtered = correlation_matrix[correlation_matrix.loc[:, correlation_matrix.columns] > .8]

correlation_matrix_filtered.head(10)

#Create dataframe just for these variables
X_pca = X.loc[:,['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment']].dropna()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

sklearn_pca = PCA(n_components=4)
Y_sklearn = sklearn_pca.fit_transform(X_pca)

print(
    'The percentage of total variance in the dataset explained by each',
    'component from Sklearn PCA.\n',
    sklearn_pca.explained_variance_ratio_)

#Recursive Feature Selection to Rank Features

# Pass any estimator to the RFE constructor
from sklearn.feature_selection import RFE

selector = RFE(rfc)
selector = selector.fit(X, Y)

print(selector.ranking_)

#Now turn into a dataframe so you can sort by rank

feature_rankings = pd.DataFrame({'Features': X.columns, 'Ranking' : selector.ranking_})
feature_rankings.sort_values('Ranking').head(100)

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = y2015.drop(['loan_status','pub_rec', 'open_acc', 'num_bc_sats', 'num_il_tl', 'total_acc', 'delinq_2yrs',
                   'avg_cur_bal', 'mort_acc', 'dti', 'total_pymnt', 'loan_amnt', 'num_sats',
               'total_bc_limit', 'inq_last_6mths', 'out_prncp'], 1)
Y = y2015['loan_status']
X = pd.get_dummies(X)
X = X.dropna(axis=1)

cross_val_score(rfc, X, Y, cv=5)

score = cross_val_score(rfc, X, Y, cv=5)

score.mean()

