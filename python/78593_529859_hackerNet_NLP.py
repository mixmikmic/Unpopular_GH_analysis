import pandas as pd

submissions = pd.read_csv("sel_hn_stories.csv")
submissions.columns = ["submission_time", "upvotes", "url", "headline"]
submissions = submissions.dropna()

submissions.head()

# Step 1: Ttokenization
"""Split each headline into individual words on the space character(" "), 
and append the resulting list to tokenized_headlines."""

tokenized_headlines = []
for item in submissions["headline"]:
    tokenized_headlines.append(item.split(" "))
    
print(tokenized_headlines[:2])    

# Step 2: Lowercasing and removing punctuation
"""For each list of tokens: Convert each individual token to lowercase
and remove all of the items from the punctuation list"""

punctuations_list = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []
for item in tokenized_headlines:
    tokens = []
    for token in item:
        token = token.lower()
        for punc in punctuations_list:
            token = token.replace(punc, "")
        tokens.append(token)
    clean_tokenized.append(tokens)

print(clean_tokenized[:2])     

# Step 3: Retrieve all of the unique words from all of the headlines
# unique_tokens contains any tokens that occur more than once across all of the headlines.

import numpy as np
unique_tokens = []
single_tokens = []
for tokens in clean_tokenized:
    for token in tokens:
        if token not in single_tokens:
            single_tokens.append(token)
        elif token in single_tokens and token not in unique_tokens:
            unique_tokens.append(token)

counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)

# Step 4: Counting Token Occurrences
for i, item in enumerate(clean_tokenized):
    for token in item:
        if token in unique_tokens:
            counts.iloc[i][token] += 1

counts.shape
counts.head(5)

word_counts = counts.sum(axis=0)
counts = counts.loc[:,(word_counts >= 5) & (word_counts <= 100)]

# Train-test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)

# Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate an instance
clf = LinearRegression()

# Fit the training data
clf.fit(X_train, y_train)

# Make predictions
y_predict = clf.predict(X_test)

mse = sum((y_predict - y_test) ** 2) / len(y_predict)
rmse = (mse)**0.5
print(rmse)

