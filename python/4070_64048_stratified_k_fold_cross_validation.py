import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

df = pd.read_csv('wine.csv', header=None)
df.head()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:].values, df.iloc[:, 0].values)

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

kfold = StratifiedKFold(y_train, n_folds=10, random_state=1, shuffle=False)
scores = []
lr = LogisticRegression(C=1.0)
for k, (train, test) in enumerate(kfold):
    lr.fit(X_train[train], y_train[train])
    score = lr.score(X_train[test], y_train[test])
    scores.append(score)
    print k + 1, np.bincount(y_train[train]), score

print np.mean(scores), np.std(scores)

scores = cross_val_score(lr, X_train, y_train, cv=10)
print scores
print np.mean(scores), np.std(scores)

