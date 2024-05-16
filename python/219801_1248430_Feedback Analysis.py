get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

#Read text file in and assign own headers

yelp_raw = pd.read_csv('yelp_labelled.txt', delimiter= '\t', header=None)
yelp_raw.columns = ['Review', 'Positive or Negative']

#Take a look at the data

yelp_raw.head(5)

yelp_raw['Review'].astype(str)

#First, make everything lower case

yelp_raw['Review'] = yelp_raw['Review'].apply(lambda x: str(x).lower())

#Extract all special characters

def GetSpecialChar(x):
    special_characters = []
    for char in x:
        if char.isalpha() == False:
            special_characters.append(char)
    return special_characters

#Create a column in the dataframe with the special characters from each row

yelp_raw['SpecialCharacters'] = yelp_raw['Review'].apply(lambda x : GetSpecialChar(x))

#Now work to get unique list

special_characters = []
for row in yelp_raw['SpecialCharacters']:
    for char in row:
        special_characters.append(char)

#Let's see our list

set(special_characters)

#Remove special characters from Review column

special_characters_list = [',', '\'', '.', '/', '"', '\'', '*', '-', '&', '%', '$', '(', ')', ':', ';', '?', '!', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


for char in special_characters_list:
        yelp_raw['Review'] = yelp_raw['Review'].str.replace(char, ' ')

#Confirm it worked
yelp_raw['Review']

#New column that splits reviews
yelp_raw['ReviewSplit'] = yelp_raw['Review'].apply(lambda x: str(x).split())

#Now get a unique count of each word in all reviews. First create 'counts' variable.

from collections import Counter
counts = yelp_raw.ReviewSplit.map(Counter).sum()

counts.most_common(100)

for count in counts:
    yelp_raw[str(count)] = yelp_raw.Review.str.contains(
        ' ' + str(count) + ' ',
        case=False)

#Correlation matrix with sns.heatmap

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(100, 100))

sns.heatmap(yelp_raw.corr())
plt.show()

#Before we actually run the model we have to build out our training data. Specify an outcome (y or dependent variable) and 
#the inputs (x or independent variables). We'll do that below under the variables data and target

data = yelp_raw.iloc[:, 4:len(yelp_raw)]
target = yelp_raw['Positive or Negative']

#Since data is binary / boolean, need to import the Bernoulli classifier.
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import RFE

# Instantiate our model and store it in a new variable.
NB_Model = BernoulliNB()

# Fit our model to the data.
NB_Model.fit(data, target)

# Classify, storing the result in a new variable.
positive_predictor = NB_Model.predict(data)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    data.shape[0],
    (target != positive_predictor).sum()))

#Confusion matrix to better understand results

from sklearn.metrics import confusion_matrix
confusion_matrix(target, positive_predictor)

#Perform Cross-Validation

from sklearn.model_selection import cross_val_score
cross_val_score(NB_Model, data, target, cv=5)

yelp_revised = yelp_raw.iloc[:, 4:len(yelp_raw.columns)]

for i in range(len(yelp_revised.columns)):
    #Create two slices and combine them
    first_slice = pd.DataFrame(yelp_revised.iloc[:, 0:i])
    second_slice = pd.DataFrame(yelp_revised.iloc[:, (i+1):len(yelp_revised.columns)])
    subset = pd.concat([first_slice, second_slice], axis=1)
    
    #Train model
    NB_Model = BernoulliNB()
    NB_Model.fit(subset, target)
    positive_predictor = NB_Model.predict(subset)
    
    #Print results for each column
    colnames = yelp_revised.columns[i]
    print("Number of mislabeled points out of a total {} points when dropping {} : {}".format(subset.shape[0], colnames, (target != positive_predictor).sum()))
    print("Accuracy {}".format(100 - ((target != positive_predictor).sum()/subset.shape[0]) * 100)) #I added this so you can view the accuracy as a percentage

# Pass any estimator to the RFE constructor
selector = RFE(NB_Model)
selector = selector.fit(data, target)

print(selector.ranking_)

#Now turn into a dataframe so you can sort by rank
rankings = pd.DataFrame({'Features': data.columns, 'Ranking' : selector.ranking_})
rankings.sort_values('Ranking').head(50)

