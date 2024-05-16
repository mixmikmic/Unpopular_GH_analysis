import os
import glob

def read_imdb_data(data_dir='data/imdb-reviews'):
    """Read IMDb movie reviews from given directory.
    
    Directory structure expected:
    - data/
        - train/
            - pos/
            - neg/
        - test/
            - pos/
            - neg/
    
    """

    # Data, labels to be returned in nested dicts matching the dir. structure
    data = {}
    labels = {}

    # Assume 2 sub-directories: train, test
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        # Assume 2 sub-directories for sentiment (label): pos, neg
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            # Fetch list of files for this sentiment
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            # Read reviews data and assign labels
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    labels[data_type][sentiment].append(sentiment)
            
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]),                     "{}/{} data size does not match labels size".format(data_type, sentiment)
    
    # Return data, labels as nested dicts
    return data, labels


data, labels = read_imdb_data()
print("IMDb reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
        len(data['train']['pos']), len(data['train']['neg']),
        len(data['test']['pos']), len(data['test']['neg'])))

print(data['train']['pos'][2])

print(data['train']['neg'][2])

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud, STOPWORDS

sentiment = 'neg'

# Combine all reviews for the desired sentiment
combined_text = " ".join([review for review in data['train'][sentiment]])

# Initialize wordcloud object
wc = WordCloud(background_color='white', max_words=50,
        # update stopwords to include common words like film and movie
        stopwords = STOPWORDS.update(['br','film','movie']))

# Generate and plot wordcloud
plt.imshow(wc.generate(combined_text))
plt.axis('off')
plt.show()

from sklearn.utils import shuffle

def prepare_imdb_data(data):
    """Prepare training and test sets from IMDb movie reviews."""
    
    # TODO: Combine positive and negative reviews and labels
    data_train   = data['train']['pos']   + data['train']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    
    data_test   = data['test']['pos']   + data['test']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    # TODO: Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train, random_state=3)
    
    data_test, labels_test   = shuffle(data_test, labels_test, random_state=3)
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


data_train, data_test, labels_train, labels_test = prepare_imdb_data(data)
print("IMDb reviews (combined): train = {}, test = {}".format(len(data_train), len(data_test)))

# BeautifulSoup to easily remove HTML tags
from bs4 import BeautifulSoup 

# RegEx for removing non-letter characters
import re

# NLTK library for the remaining steps
import nltk
#nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords # import stopwords

from nltk.stem.porter import *
stemmer = PorterStemmer()

def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    
    # TODO: Remove HTML tags and non-letters,
    #       convert to lowercase, tokenize,
    #       remove stopwords and stem

    # Return final list of words
    
    #1- remove HTML tags
    soup   = BeautifulSoup(review, "html5lib")
    review = soup.get_text()
    
    #2 remove punctuations and none-letters
    review = re.sub(r"[^a-zA-z0-9]", " ", review)
    
    #3- lower case
    review = review.lower()
    
    #4- tokenize
    words = review.split()
    
    #5- remove stop words
    words = [w.strip() for w in words if w not in stopwords.words("english")]
    
    #6- stem
    words = [stemmer.stem(w) for w in words]
    
    return words


review_to_words("""This is just a <em>test</em>.<br/><br />
But if it wasn't a test, it would make for a <b>Great</b> movie review!""")

import nltk

print('The nltk version is {}.'.format(nltk.__version__))

import pickle

cache_dir = os.path.join("cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        words_train = list(map(review_to_words, data_train))
        words_test = list(map(review_to_words, data_test))
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test


# Preprocess data
words_train, words_test, labels_train, labels_test = preprocess_data(
        data_train, data_test, labels_train, labels_test)

# Take a look at a sample
print("\n--- Raw review ---")
print(data_train[1])
print("\n--- Preprocessed words ---")
print(words_train[1])
print("\n--- Label ---")
print(labels_train[1])

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
# joblib is an enhanced version of pickle that is more efficient for storing NumPy arrays

def extract_BoW_features(words_train, words_test, vocabulary_size=5000,
                         cache_dir=cache_dir, cache_file="bow_features.pkl"):
    """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""
    
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # TODO: Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer     = CountVectorizer(preprocessor=lambda x: x, 
                                         tokenizer=lambda x: x, 
                                         max_features=vocabulary_size)
        features_train = vectorizer.fit_transform(words_train).toarray()

        # TODO: Apply the same vectorizer to transform the test documents (ignore unknown words)
        features_test  = vectorizer.fit_transform(words_train).toarray()
        
        # NOTE: Remember to convert the features using .toarray() for a compact representation
        
        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test,
                             vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                cache_data['features_test'], cache_data['vocabulary'])
    
    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary


# Extract Bag of Words features for both training and test datasets
features_train, features_test, vocabulary = extract_BoW_features(words_train, words_test)

# Inspect the vocabulary that was computed
print("Vocabulary: {} words".format(len(vocabulary)))

import random
print("Sample words: {}".format(random.sample(list(vocabulary.keys()), 8)))

# Sample
print("\n--- Preprocessed words ---")
print(words_train[5])
print("\n--- Bag-of-Words features ---")
print(features_train[5])
print("\n--- Label ---")
print(labels_train[5])

# Plot the BoW feature vector for a training document
x = np.array([[1,2,3],
     [4,5,6],
     [7,8,9]])
print(x[2])

plt.plot(features_train[5,:])
plt.xlabel('Word')
plt.ylabel('Count')
plt.show()

train_size     = features_train.shape[0] * features_train.shape[1]
non_zero_size  = np.count_nonzero(features_train)
zero_size      = (train_size-non_zero_size)
print('training data size: ', train_size)
print('zero items count: '  , (train_size-non_zero_size))
print('none-zero items count: '  , non_zero_size)

print('zero %{}: '.format( ((zero_size/train_size)*100)) )

# Find number of occurrences for each word in the training set
word_freq = features_train.sum(axis=0)

# Sort it in descending order
sorted_word_freq = np.sort(word_freq)[::-1]

# Plot 
plt.plot(sorted_word_freq)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel('Rank')
plt.ylabel('Number of occurrences')
plt.show()

print(sorted_word_freq[:5])

import sklearn.preprocessing as pr

# TODO: Normalize BoW features in training and test set
features_train = pr.normalize(features_train, copy=False)
features_test  = pr.normalize(features_test, copy=False)

print("features after normalize: ", features_train[2])

from sklearn.naive_bayes import GaussianNB

# TODO: Train a Guassian Naive Bayes classifier
clf1 = GaussianNB().fit(features_train, labels_train)

# Calculate the mean accuracy score on training and test sets
print("[{}] Accuracy: train = {}, test = {}".format(
        clf1.__class__.__name__,
        clf1.score(features_train, labels_train),
        clf1.score(features_test, labels_test)))

#Custom Code: Test GaussianNB 
print("real value: ", labels_train[:100])
#1-predect feature
print("predection: ", clf1.predict(features_train[:100]))

from sklearn.ensemble import GradientBoostingClassifier

#Testing 1: no model selection
#learning rate = [2-10]/trees.

#testing 1: 100 is slowe --> reduce to 80
n_estimators = 80

def classify_gboost(X_train, X_test, y_train, y_test):        
    # Initialize classifier
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, max_depth=1, random_state=0)

    # TODO: Classify the data using GradientBoostingClassifier
    clf.fit(X_train, y_train)
    
    # TODO(optional): Perform hyperparameter tuning / model selection
    
    # TODO: Print final training & test accuracy
    print("[{}] Accuracy: train = {}, test = {}".format(
        clf.__class__.__name__,
        clf.score(X_train, y_train),
        clf.score(X_test, y_test)))
    
    # Return best classifier model
    return clf


clf2 = classify_gboost(features_train, features_test, labels_train, labels_test)

#test range result
print(list([30,50,70,80, 90, 100]))

from sklearn.grid_search import GridSearchCV

#Testing 2: with model selection
'''reference: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/'''
def classify_gboost2(X_train, X_test, y_train, y_test):        
    # Initialize classifier
    clf_temp = GradientBoostingClassifier(learning_rate=1.0, max_depth=1, random_state=0)
    
    # TODO(optional): Perform hyperparameter tuning / model selection
    esitimator_params = {'n_estimators':list([30,70, 80, 90])}
    gsearch = GridSearchCV(estimator=clf_temp,param_grid=esitimator_params,verbose=2,n_jobs=4)
    
    # TODO: Classify the data using GradientBoostingClassifier
    gsearch.fit(X_train, y_train)
    clf = gsearch.best_estimator_
    
    # TODO: Print final training & test accuracy
    print("best params: ")
    print(gsearch.best_params_)
    print("best score: ")
    print(gsearch.best_score_)
    print("----------------")
    print("[{}] Accuracy: train = {}, test = {}".format(
        clf.__class__.__name__,
        clf.score(X_train, y_train),
        clf.score(X_test, y_test)))
    
    # Return best classifier model
    return clf


clf2 = classify_gboost2(features_train, features_test, labels_train, labels_test)

# TODO: Write a sample review and set its true sentiment
my_review = "Although the main actor was good in his role the overall movie was boring and doesn't have story"
true_sentiment = 'neg'  # sentiment must be 'pos' or 'neg'

# TODO: Apply the same preprocessing and vectorizing steps as you did for your training data

#1- clear text and split to words
my_words = list(map(review_to_words, [my_review]))

print('words: ', my_words)

#2- compute bag of words features
vectorizer  = CountVectorizer(vocabulary=vocabulary,preprocessor=lambda x: x,tokenizer=lambda x: x)
features_review = vectorizer.fit_transform(my_words).toarray()

#3- normalize features
features_review = pr.normalize(features_review, copy=False)

print('features shape: ', features_review.shape)

# TODO: Then call your classifier to label it
print("predection 1: ", clf1.predict(features_review)[0] )
print("predection 2: ", clf2.predict(features_review)[0] )
print("predection real: ",true_sentiment)

from keras.datasets import imdb  # import the built-in imdb dataset in Keras

# Set the vocabulary size
vocabulary_size = 5000

# Load in training and test data (note the difference in convention compared to scikit-learn)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))

# Inspect a sample review and its label
print("--- Review ---")
print(X_train[5])
print("--- Label ---")
print(y_train[5])

# Map word IDs back to words
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print("--- Review (with words) ---")
print([id2word.get(i, " ") for i in X_train[5]])
print("--- Label ---")
print(y_train[5])

X_train.shape

from keras.preprocessing import sequence

# Set the maximum number of words per document (for both training and testing)
max_words = 500

# TODO: Pad sequences in X_train and X_test
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test  = sequence.pad_sequences(X_test, maxlen=max_words)

print(X_train.shape)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# TODO: Design your model
model = Sequential()
#embedding reference: https://keras.io/layers/embeddings/
model.add(Embedding(vocabulary_size, 128))

model.add(LSTM(128))

model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# TODO: Compile your model, specifying a loss function, optimizer, and metrics
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# TODO: Specify training parameters: batch size and number of epochs
batch_size = 32 #Keras cheatsheet recommend this
num_epochs = 15 

# TODO(optional): Reserve/specify some training data for validation (not to be used for training)
split_index = (int)(X_train.shape[0] * 0.8)
X_train1, X_validate = X_train[:split_index], X_train[split_index:]
y_train1, y_validate = y_train[:split_index], y_train[split_index:]

print('train size: ', X_train1.shape)
print('validate size: ', X_validate.shape)

# TODO: Train your model
model_file = "rnn_model.h5"  # HDF5 file
if os.path.exists(os.path.join(cache_dir, model_file) == False):
    model.fit(X_train1, y_train1,
              batch_size=batch_size, 
              epochs=num_epochs, 
              verbose=1, 
              validation_data=(X_validate, y_validate))

# Save your model, so that you can quickly load it in future (and perhaps resume training)
model.save(os.path.join(cache_dir, model_file))

# Later you can load it using keras.models.load_model()
from keras.models import load_model
model = load_model(os.path.join(cache_dir, model_file))

# Evaluate your model on the test set
scores = model.evaluate(X_test, y_test, verbose=0)  # returns loss and other metrics specified in model.compile()
print("Test accuracy:", scores[1])  # scores[1] should correspond to accuracy if you passed in metrics=['accuracy']

# Evaluate your model on the train set
scores = model.evaluate(X_train, y_train, verbose=0)  # returns loss and other metrics specified in model.compile()
print("Train accuracy:", scores[1])  # scores[1] should correspond to accuracy if you passed in metrics=['accuracy']



