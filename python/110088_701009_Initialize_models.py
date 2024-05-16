# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import read_df, remove_links, clean_sentence, save_object, load_object

import os
directories = ['objects', 'models', 'clusters', 'matricies']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

model_name = "example_model"

# Get the data from the csv
df = read_df('data',extension = "/*.csv")

# Do an inspection of our data to ensure nothing went wrong
df.info()

df.head()

# Clean the text in the dataframe
df = df.replace(np.nan, '', regex = True)
df = df.replace("\[deleted\]", '', regex = True)
df["rawtext"] = df["title"] + " " + df["selftext"]
df["cleantext"] = df["rawtext"].apply(remove_links).apply(clean_sentence)

# Check that the cleaning was successful
df.info()

df.head()

# Get a stream of tokens
posts = df["cleantext"].apply(lambda str: str.split()).tolist()

# Train a phraseDetector to join two word phrases together
two_word_phrases = gensim.models.Phrases(posts)
two_word_phraser = gensim.models.phrases.Phraser(two_word_phrases)

# Train a phraseDetector to join three word phrases together
three_word_phrases = gensim.models.Phrases(two_word_phraser[posts])
three_word_phraser = gensim.models.phrases.Phraser(three_word_phrases)
posts = list(three_word_phraser[two_word_phraser[posts]])

# Update Data frame
df["phrasetext"] = df["cleantext"].apply(lambda str: " ".join(three_word_phraser[two_word_phraser[str.split()]]))

# Ensure posts contain same number of elements
len(posts) == len(df)

# Check that the dataframe was updated correctly
for i in range(len(posts)):
    if not " ".join(posts[i]) == list(df["phrasetext"])[i]:
        print("index :" + str(i) + " is incorrect")

save_object(posts, 'objects/', model_name + "-posts")
save_object(df, 'objects/', model_name + "-df")

# Set the minimum word count to 10. This removes all words that appear less than 10 times in the data
minimum_word_count = 10
# Set skip gram to 1. This sets gensim to use the skip gram model instead of the Continuous Bag of Words model
skip_gram = 1
# Set Hidden layer size to 300.
hidden_layer_size = 300
# Set the window size to 5. 
window_size = 5
# Set hierarchical softmax to 1. This sets gensim to use hierarchical softmax
hierarchical_softmax = 1
# Set negative sampling to 20. This is good for relatively small data sets, but becomes harder for larger datasets
negative_sampling = 20

# Build the model
model = gensim.models.Word2Vec(posts, min_count = minimum_word_count, sg = skip_gram, size = hidden_layer_size,
                               window = window_size, hs = hierarchical_softmax, negative = negative_sampling)

model.most_similar(positive = ["kitten"])

model.most_similar(positive = ["father", "woman"], negative = ["man"])

model.most_similar(positive = ["family", "obligation"], negative = ["love"])

model.save('models/' + model_name + '.model')
del model

model = gensim.models.Word2Vec.load('models/' + model_name + '.model')

# Initialize the list of words used
vocab_list = sorted(list(model.wv.vocab))

# Extract the word vectors
vecs = []
for word in vocab_list:
    vecs.append(model.wv[word].tolist())

# change array format into numpy array
WordsByFeatures = np.array(vecs)

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s:s), lst))), min_df = 0)

# Make Posts By Words Matrix
PostsByWords = countvec.fit_transform(posts)

# Check that PostsByWords is the number of Posts by the number of words
PostsByWords.shape[0] == len(posts)

# check that the number of words is consistant for all matricies
PostsByWords.shape[1] == len(WordsByFeatures)

save_object(PostsByWords,'matricies/', model_name + "-PostsByWords")
save_object(WordsByFeatures,'matricies/', model_name + "-WordsByFeatures")

from sklearn.cluster import KMeans
# get the fit for different values of K
test_points = [12] + list(range(25, 401, 25))
fit = []
for point in test_points:
    kmeans = KMeans(n_clusters = point, random_state = 42).fit(WordsByFeatures)
    save_object(kmeans, 'clusters/', model_name + "-words-cluster_model-" + str(point))
    fit.append(kmeans.inertia_)

save_object(fit, 'objects/', model_name + "-words" + "-fit")
save_object(test_points, 'objects/', model_name + "-words" + "-test_points")
del fit
del test_points

