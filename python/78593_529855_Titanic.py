# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Loading data and printing first few rows
titanic_DF = pd.read_csv('train.csv')
test_DF = pd.read_csv('test.csv')

titanic_DF.head(2)

# Similarly look into test data 
test_DF.head(2)

# Previewing the statistics of training data and test data
titanic_DF.info()
# print('')
# test_DF.info()

# Data Visualization 
plt.rc('font', size=24)
fig = plt.figure(figsize=(18, 8))
alpha = 0.6

# Plot pclass distribution
ax1 = plt.subplot2grid((2,3), (0,0))
titanic_DF.Pclass.value_counts().plot(kind='barh', color='blue', label='train', alpha=alpha)
test_DF.Pclass.value_counts().plot(kind='barh',color='magenta', label='test', alpha=alpha)
ax1.set_ylabel('Pclass')
ax1.set_xlabel('Frequency')
ax1.set_title("Distribution of Pclass" )
plt.legend(loc='best')

# Plot sex distribution
ax2 = plt.subplot2grid((2,3), (0,1))
titanic_DF.Sex.value_counts().plot(kind='barh', color='blue', label='train', alpha=alpha)
test_DF.Sex.value_counts().plot(kind='barh', color='magenta', label='test', alpha=alpha)
ax2.set_ylabel('Sex')
ax2.set_xlabel('Frequency')
ax2.set_title("Distribution of Sex" )
plt.legend(loc='best')


# Plot Embarked Distribution
ax5 = plt.subplot2grid((2,3), (0,2))
titanic_DF.Embarked.fillna('S').value_counts().plot(kind='barh', color='blue', label='train', alpha=alpha)
test_DF.Embarked.fillna('S').value_counts().plot(kind='barh',color='magenta', label='test', alpha=alpha)
ax5.set_ylabel('Embarked')
ax5.set_xlabel('Frequency')
ax5.set_title("Distribution of Embarked" )
plt.legend(loc='best')

# Plot Age distribution
ax3 = plt.subplot2grid((2,3), (1,0))
titanic_DF.Age.fillna(titanic_DF.Age.median()).plot(kind='kde', color='blue', label='train', alpha=alpha)
test_DF.Age.fillna(test_DF.Age.median()).plot(kind='kde',color='magenta', label='test', alpha=alpha)
ax3.set_xlabel('Age')
ax3.set_title("Distribution of age" )
plt.legend(loc='best')

# Plot fare distribution
ax4 = plt.subplot2grid((2,3), (1,1))
titanic_DF.Fare.fillna(titanic_DF.Fare.median()).plot(kind='kde', color='blue', label='train', alpha=alpha)
test_DF.Fare.fillna(test_DF.Fare.median()).plot(kind='kde',color='magenta', label='test', alpha=alpha)
ax4.set_xlabel('Fare')
ax4.set_title("Distribution of Fare" )
plt.legend(loc='best')

plt.tight_layout()


# print the names of the columns in the data frame
titanic_DF.columns
# Check which columns have missing data
for column in titanic_DF.columns:
    if np.any(pd.isnull(titanic_DF[column])) == True:
        print(column)

# Plot pclass distribution
fig = plt.figure(figsize=(6, 6))

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_DF, size=4,color="green")
plt.ylabel('Fraction Survived')
plt.xlabel('Pclass')
plt.title("Survival according to Class" )

# Plot Gender Survival
fig = plt.figure(figsize=(6,6))
sns.factorplot('Sex','Survived', data=titanic_DF, size=4,color="green")
plt.ylabel('Fraction Survived')
plt.xlabel('Gender')
plt.title("Survival according to Gender" )

# Plot Fare
fig = plt.figure(figsize=(15, 6))
titanic_DF[titanic_DF.Survived==0].Fare.plot(kind='density', color='red', label='Died', alpha=alpha)
titanic_DF[titanic_DF.Survived==1].Fare.plot(kind='density',color='green', label='Survived', alpha=alpha)
plt.ylabel('Density')
plt.xlabel('Fare')
plt.xlim([-100,200])
plt.title("Distribution of Fare for Survived and Did not survive" )

plt.legend(loc='best')
plt.grid()

# Filling missing age data with median values
titanic_DF["Age"] = titanic_DF["Age"].fillna(titanic_DF["Age"].median())
titanic_DF.describe()

# Plot age
fig = plt.figure(figsize=(15, 6))
titanic_DF[titanic_DF.Survived==0].Age.plot(kind='density', color='red', label='Died', alpha=alpha)
titanic_DF[titanic_DF.Survived==1].Age.plot(kind='density',color='green', label='Survived', alpha=alpha)
plt.ylabel('Density')
plt.xlabel('Age')
plt.xlim([-10,90])
plt.title("Distribution of Age for Survived and Did not survive" )
plt.legend(loc='best')
plt.grid()

# data cleaning for Embarked
print (titanic_DF["Embarked"].unique())
print (titanic_DF.Embarked.value_counts())

# filling Embarked data with most frequent 'S'
titanic_DF["Embarked"] = titanic_DF["Embarked"].fillna('S')
titanic_DF.loc[titanic_DF["Embarked"] == 'S', "Embarked"] = 0
titanic_DF.loc[titanic_DF["Embarked"] == 'C', "Embarked"] = 1
titanic_DF.loc[titanic_DF["Embarked"] == 'Q', "Embarked"] = 2

# convert female/male to numeric values (male=0, female=1)
titanic_DF.loc[titanic_DF["Sex"]=="male","Sex"]=0
titanic_DF.loc[titanic_DF["Sex"]=="female","Sex"]=1
titanic_DF.head(5)

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

# columns we'll use to predict outcome
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]


# instantiate the model
logreg = LogisticRegression()

# perform cross-validation
print(cross_val_score(logreg, titanic_DF[predictors], titanic_DF['Survived'], cv=10, scoring='accuracy').mean())

# print the names of the columns in the data frame
test_DF.columns
# Check which columns have missing data
for column in test_DF.columns:
    if np.any(pd.isnull(test_DF[column])) == True:
        print(column)

# Filling missing age data with median values
test_DF["Age"] = test_DF["Age"].fillna(titanic_DF["Age"].median())

# filling Embarked data with most frequent 'S'
test_DF["Embarked"] = test_DF["Embarked"].fillna('S')
test_DF.loc[test_DF["Embarked"] == 'S', "Embarked"] = 0
test_DF.loc[test_DF["Embarked"] == 'C', "Embarked"] = 1
test_DF.loc[test_DF["Embarked"] == 'Q', "Embarked"] = 2

# convert female/male to numeric values (male=0, female=1)
test_DF.loc[test_DF["Sex"]=="male","Sex"]=0
test_DF.loc[test_DF["Sex"]=="female","Sex"]=1

test_DF.describe()

# Test also has empty fare columns
test_DF["Fare"] = test_DF["Fare"].fillna(test_DF["Fare"].median())

# Apply our prediction to test data
logreg.fit(titanic_DF[predictors], titanic_DF["Survived"])
prediction = logreg.predict(test_DF[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset
submission_DF = pd.DataFrame({ 
    "PassengerId" : test_DF["PassengerId"],
    "Survived" : prediction
    })
print(submission_DF.head(5))

# prepare file for submission
submission_DF.to_csv("submission.csv", index=False)



