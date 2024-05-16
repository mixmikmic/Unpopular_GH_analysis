#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim
from sklearn.cluster import KMeans

# Import utility files
from utils import save_object, load_object, make_post_clusters, make_clustering_objects

from orangecontrib.associate.fpgrowth import *

# Set the model we are going to be analyzing
model_name = "PTSD_model"

# Initialize a word clustering to use
num_word_clusters = 100
# Initialize the threshold to count a correlation
correlation_threshold = 0.65

df = load_object('objects/', model_name + '-df')

# Load Our Saved matricies
PostsByWords = load_object('matricies/', model_name + "-PostsByWords")
WordsByFeatures = load_object('matricies/', model_name + "-WordsByFeatures")

# Generate the posts by Features matrix through matrix multiplication
PostsByFeatures = PostsByWords.dot(WordsByFeatures)
PostsByFeatures = np.matrix(PostsByFeatures)
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')

vocab_list = sorted(list(model.wv.vocab))

kmeans =  load_object('clusters/', model_name + '-words-cluster_model-' + str(num_word_clusters))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

clusterWords = list(map(lambda x: list(map(lambda y: y[0] , x["word_list"])), clusters))

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s: s), lst))), min_df = 0)

# Make Clusters By Words Matrix
ClustersByWords = countvec.fit_transform(clusterWords)

# take the transpose of Clusters
WordsByCluster = ClustersByWords.transpose()

# Multiply Posts by Words by Words By cluster to get Posts By cluster
PostsByClusters = PostsByWords.dot(WordsByCluster)

X = np.array(PostsByClusters.todense())

cluster_df = pd.DataFrame(data = X)

correlations = cluster_df.corr().values

# Sort all the words in the words list
for cluster in clusters:
    cluster["word_list"].sort(key = lambda x:x[1], reverse = True)

correlations_list = []
for i in range(len(correlations)):
    for j in range(i+1,len(correlations[0])):
        corr_val = correlations[i][j]
        if corr_val > correlation_threshold:
            correlations_list.append([i,j,corr_val,clusters[i]["word_list"][:5],clusters[j]["word_list"][:5]])

len(correlations_list)

correlations_list

import os
directories = ['correlation-analysis']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

import csv
heading = ["cluster 1 number", "cluster 2 number", "correlation values","cluster 1","cluster 2"]
with open("correlation-analysis/"+model_name+"-correlations.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(heading)
    [writer.writerow(r) for r in correlations_list]

