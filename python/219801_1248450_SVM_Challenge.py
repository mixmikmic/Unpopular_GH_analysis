import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')

raw_data = pd.read_csv('epi_r.csv')

raw_data.shape

list(raw_data.columns)

raw_data.rating.describe()

raw_data.rating.hist(bins=20)
plt.title('Histogram of Recipe Ratings')
plt.show()

# Count nulls 
null_count = raw_data.isnull().sum()
null_count[null_count>0]

# svr = SVR()
# X = raw_data.drop(['rating', 'title', 'calories', 'protein', 'fat', 'sodium'], 1)
# Y = raw_data.rating
# svr.fit(X,Y)

# plt.scatter(Y, svr.predict(X))

# svr.score(X, Y)

# cross_val_score(svr, X, Y, cv=5)

#1 would equal 'Good' rating, 0 would equal 'Bad' rating
raw_data['rating'] = np.where((raw_data['rating'] >= 3), 1, 0)

rating_count = raw_data['rating'].sum()
print('Baseline accuracy for Rating is: ' + str(round((rating_count/raw_data.shape[0])*100, 2)) + '%')

#First, instantiate model and fit our data

X = raw_data.drop(['rating', 'title', 'calories', 'protein', 'fat', 'sodium'], 1)
Y = raw_data.rating

svm = SVC()
svm.fit(X, Y)

# Pass SVM model to the RFE constructor
from sklearn.feature_selection import RFE

selector = RFE(svm)
selector = selector.fit(X, Y)

#Now turn results into a dataframe so you can sort by rank

rankings = pd.DataFrame({'Features': va_crime_features.columns, 'Ranking' : selector.ranking_})
rankings.sort_values('Ranking').head(30)

