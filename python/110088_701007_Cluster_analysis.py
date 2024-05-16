#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object, make_clustering_objects

import os
directories = ['cluster-analysis']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# Set the model we are going to be analyzing
model_name = "example_model"

# Load the fit and test point values
fit = load_object('objects/', model_name + "-words" + "-fit")
test_points = load_object('objects/', model_name + "-words" + "-test_points")

# Plot the fit for each size
plt.plot(test_points, fit, 'ro')
plt.axis([0, 400, 0, np.ceil(fit[0] + (1/10)*fit[0])])
plt.show()

# Set the number of clusters to analyze
num_clusters = 100 

# load the models
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')
kmeans = load_object('clusters/', model_name + "-words-cluster_model-" + str(num_clusters))
WordsByFeatures = load_object('matricies/', model_name + '-' + 'WordsByFeatures')

vocab_list = sorted(list(model.wv.vocab))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

# Sort all the words in the words list
for cluster in clusters:
    cluster["word_list"].sort(key = lambda x:x[1], reverse = True)

# Set the number of words to display. The table with contain the top size_words_list words
size_words_list = 100
table = []
for i in range(len(clusters)):
    row = []
    row.append("cluster " + str(i+1))
    row.append(clusters[i]["total_freq"])
    row.append(clusters[i]["unique_words"])
    for j in range(size_words_list):
        try:
            row.append(clusters[i]["word_list"][j])
        except:
            break
    table.append(row)

import csv
with open('cluster-analysis/' + model_name + "-" + str(num_clusters) + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    [writer.writerow(r) for r in table]

# Fit the model to the clusters
from sklearn.manifold import MDS
mds = MDS().fit(kmeans.cluster_centers_)

# Get the embeddings
embedding = mds.embedding_.tolist()
x = list(map(lambda x:x[0], embedding))
y = list(map(lambda x:x[1], embedding))

top_words= list(map(lambda x: x[0][0], map(lambda x: x["word_list"], clusters)))

# Plot the Graph with top words
plt.figure(figsize = (20, 10))
plt.plot(x, y, 'bo')
for i in range(len(top_words)):
    plt.annotate(top_words[i], (x[i], y[i]))
plt.show()

