get_ipython().magic('matplotlib inline')

import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

sns.set_style('whitegrid')

wine_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

print("The 'wine' dataframe has {0} rows and {1} columns".format(*wine_df.shape))
print("\nColumn names are:\n\t"+'\n\t'.join(wine_df.columns))
print("\nFirst 5 rows of the dataframe are:")
wine_df.head()

y = wine_df.quality.values
X = wine_df.ix[:,wine_df.columns != "quality"].as_matrix()

plt.hist(y,20)
plt.xlabel('quality')
plt.ylabel('# of records')
plt.title('Distribution of qualities');

y = [1 if i >=7 else 0 for i in y]

print('Class 0 - {:.3f} %'.format(100*y.count(0)/len(y)))
print('Class 1 - {:.3f} %'.format(100*y.count(1)/len(y)))



scores = []

for val in range(1,41):
    clf = RandomForestClassifier(n_estimators = val)
    validated = cross_val_score(clf, X, y, cv = 10)
    scores.append(validated)
    
validated

scores_per_fold = pd.DataFrame(scores).transpose()
print(scores_per_fold.shape)

sns.boxplot(scores_per_fold);
plt.xlabel('number of trees')
plt.show()

scores_per_fold.head()



