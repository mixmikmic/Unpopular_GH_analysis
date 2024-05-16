# Importing essential Python libraries

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

titanic_df = pd.read_csv('train.csv')

#Let's take a preview of the data
titanic_df.head()

#we're missing a lot of cabin info
titanic_df.info()

#1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)

# Let's first check gender
sns.factorplot('Sex',data=titanic_df,kind='count')

# Now let's seperate the genders by classes, remember we can use the 'hue' arguement here!

sns.factorplot('Sex',data=titanic_df,hue='Pclass', kind='count')
sns.factorplot('Pclass',data=titanic_df,hue='Sex', kind='count')

# We'll treat anyone as under 16 as a child
#  a function to sort through the sex 
def male_female_child(passenger):
    # Take the Age and Sex
    age,sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex

titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)

titanic_df.head(10)

# Let's try the factorplot again!
sns.factorplot('Pclass',data=titanic_df,hue='person', kind='count')

#age : histogram using pandas
titanic_df['Age'].hist(bins=70)

#Mean age of passengers
titanic_df['Age'].mean()

titanic_df['person'].value_counts()

fig = sns.FacetGrid(titanic_df, hue="Sex",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df, hue="person",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# Let's do the same for class by changing the hue argument:
fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

#Dropping null values
deck = titanic_df['Cabin'].dropna()

deck.head()

#Notice we only need the first letter of the deck to classify its level (e.g. A,B,C,D,E,F,G)

levels = []

# Loop to grab first letter
for level in deck:
    levels.append(level[0])    

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d', kind= 'count', order=['A','B','C','D','E','F'])

#Note here that the Embarked column has C,Q,and S values. 
#Reading about the project on Kaggle you'll note that these stand for Cherbourg, Queenstown, Southhampton.


sns.factorplot('Embarked',data=titanic_df,hue='Pclass',x_order=['C','Q','S'], kind = 'count')

# Let's start by adding a new column to define alone

# We'll add the parent/child column with the sibsp column
titanic_df['Alone'] =  titanic_df.Parch + titanic_df.SibSp

# Look for >0 or ==0 to set alone status
titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

titanic_df['Alone'].head()

sns.factorplot('Alone',hue= 'Pclass',data=titanic_df,palette='Blues', kind = 'count')
sns.factorplot('Pclass',hue= 'Alone',data=titanic_df,order= [1,2,3], palette='Blues', kind = 'count')

# Let's start by creating a new column for legibility purposes through mapping (Lec 36)
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

# Let's just get a quick overall view of survied vs died. 
sns.factorplot('Survivor',data=titanic_df,palette='Set1', kind = 'count')

# Let's use a factor plot again, but now considering class
sns.factorplot('Pclass','Survived',data=titanic_df, order = [1,2,3])

# Let's use a factor plot again, but now considering class and gender
sns.factorplot('Pclass','Survived',hue='person',data=titanic_df,order = [1,2,3])

# Let's use a linear plot on age versus survival
sns.lmplot('Age','Survived',data=titanic_df, hue='Pclass',palette='winter')

# Let's use a linear plot on age versus survival using hue for class seperation
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)

sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)


levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']

cabin_df = cabin_df[cabin_df.Cabin != 'T']
titanic_df['Level'] = Series(levels,index=deck.index)

sns.factorplot('Level','Survived',x_order=['A','B','C','D','E','F'],data=titanic_df)

sns.factorplot('Alone','Survived',data=titanic_df)



