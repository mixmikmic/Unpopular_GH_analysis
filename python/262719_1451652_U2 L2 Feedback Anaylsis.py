import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')

# Start with amazon review first.
amazon_reviews = pd.read_csv('amazon_cells_labelled.txt', delimiter='\t')

# If you run .head() without doing anything, we will only see two columns, 
# comments and a number ( 0 or 1). 
# lets look only at positive reviews.
amazon_reviews.columns = ['reviews', 'positive']
amazon_reviews['positive'] = (amazon_reviews['positive'] == 1)

print(amazon_reviews.head())

# Lets take a look at the data now and see if I can find any words to use for the keywords. 
print(amazon_reviews.head(50))

keywords = ['good', 'excellent','great', 'nice', 'awesome', 'fantastic',
           'well', 'wonderful', 'ideal', 'quick', 'best', 'fair']

for key in keywords:
    amazon_reviews[str(key)] = amazon_reviews.reviews.str.contains(
        str(key),
        case=False
    )

# Time to check if the keywords are correlated; if not, then proceed with the Bernoulli NB
sns.heatmap(amazon_reviews.corr())

data = amazon_reviews[keywords]
target = amazon_reviews['positive']

# Our data is binary / boolean, so we're importing the Bernoulli classifier.
from sklearn.naive_bayes import BernoulliNB

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of positive reviews out of a total {} reviews : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))

# Try this classifier on IMDB.

imdb_reviews = pd.read_csv('imdb_labelled.txt', delimiter='\t')

# Same thing as before, need to add column names to both columns.
imdb_reviews.columns = ['reviews', 'positive']
imdb_reviews['positive'] = (imdb_reviews['positive'] == 1)

print(imdb_reviews.head(50))

# I would change some of the keywords, but I'm supposed to test how well
# this model works on other datasets. 
keywords = ['good', 'excellent','great', 'nice', 'awesome', 'fantastic',
           'well', 'wonderful', 'ideal', 'quick', 'best', 'fair']

for key in keywords:
    imdb_reviews[str(key)] = imdb_reviews.reviews.str.contains(
        str(key),
        case=False
    )

# Time to check if the keywords are correlated; if not, then proceed with the Bernoulli NB
sns.heatmap(imdb_reviews.corr())

data = imdb_reviews[keywords]
target = imdb_reviews['positive']

# Our data is binary / boolean, so we're importing the Bernoulli classifier.
from sklearn.naive_bayes import BernoulliNB

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of positive reviews out of a total {} reviews : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))



