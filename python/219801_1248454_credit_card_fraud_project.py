import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import ensemble
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')

fraud = pd.read_csv('creditcard.csv')

fraud.shape

fraud.head(5)

fraud.columns

fraud.describe()

#First look at Time
sns.distplot(fraud.Time)
plt.title('Distribution of Time')
plt.show()

#Now look at Amount
sns.boxplot(x=fraud['Amount'])
plt.title('Distribution of Amount')
plt.show()

fraud_total = fraud['Class'].sum()
print('Baseline accuracy for fraud is: ' + str(round((fraud_total/fraud.shape[0])*100, 2)) + '%')

#Set up our independent variables and outcome variable

X = fraud.iloc[:,0:30]
Y = fraud.Class

#Setup function to run our model with different k parameters

def KNN_Model(k):
    KNN = KNeighborsClassifier(n_neighbors=k, weights='distance')
    KNN.fit(X, Y)
    print('\n Percentage accuracy for K Nearest Neighbors Classifier')
    print(str(KNN.score(X, Y)*100) + '%')
    print(cross_val_score(KNN, X, Y, cv=10))

#Run the model with K=10
KNN_Model(10)

#Set up function to run our model with different trees, criterion, max features and max depth

def RFC_Model(trees, criteria, num_features, depth):
    rfc = ensemble.RandomForestClassifier(n_estimators=trees, criterion=criteria, max_features=num_features, max_depth=depth)
    rfc.fit(X, Y)
    print('\n Percentage accuracy for Random Forest Classifier')
    print(str(rfc.score(X, Y)*100) + '%')
    print(cross_val_score(rfc, X, Y, cv=10))

#Run the model with 50 trees, criterion = 'entropy', max features = 5 and max depth = 5
RFC_Model(50, 'entropy', 5, 5)

#Try RFC again, same parameters accept use 'gini' instead of 'entropy' for criterion
RFC_Model(50, 'gini', 5, 5)

#Set up function to run our model using lasso or ridge regularization and specifying alpha
#parameter

def Logistic_Reg_Model(regularization, alpha):
    lr = LogisticRegression(penalty=regularization, C=alpha)
    lr.fit(X, Y)
    print('\n Percentage accuracy for Logistic Regression')
    print(str(lr.score(X, Y)*100) + '%')
    print(cross_val_score(lr, X, Y, cv=10))

#Run using 'l1' (lasso) penalty and 0.8 alpha
Logistic_Reg_Model('l1', 0.8)

#Run using 'l2' (ridge) penalty and 100 alpha
Logistic_Reg_Model('l2', 100)

svm = SVC()
svm.fit(X, Y)
print(str(svm.score(X, Y)*100) + '%')



