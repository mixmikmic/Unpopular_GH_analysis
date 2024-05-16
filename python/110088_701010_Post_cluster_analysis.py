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

import os
directories = ['post-analysis']
for dirname in directories:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# Set the model we are going to be analyzing
model_name = "model6"

df = load_object('objects/', model_name + '-df')

scores = list(df['score'])
num_comments_list = list(df['num_comments'])

# Load Our Saved matricies
PostsByWords = load_object('matricies/', model_name + "-PostsByWords")
WordsByFeatures = load_object('matricies/', model_name + "-WordsByFeatures")

# Generate the posts by Features matrix through matrix multiplication
PostsByFeatures = PostsByWords.dot(WordsByFeatures)
PostsByFeatures = np.matrix(PostsByFeatures)
len(PostsByFeatures)

model = gensim.models.Word2Vec.load('models/' + model_name + '.model')

vocab_list = sorted(list(model.wv.vocab))

# Initialize a word clustering to use
num_word_clusters = 100
kmeans =  load_object('clusters/', model_name + '-words-cluster_model-' + str(num_word_clusters))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

clusterWords = list(map(lambda x: list(map(lambda y: y[0] , x["word_list"])), clusters))

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer(vocabulary = vocab_list, analyzer = (lambda lst:list(map((lambda s: s), lst))), min_df = 0)

# Make Clusters By Words Matrix
ClustersByWords = countvec.fit_transform(clusterWords)

# Ensure consistency
len(WordsByFeatures) == ClustersByWords.shape[1]

# take the transpose of Clusters
WordsByCluster = ClustersByWords.transpose()

# Multiply Posts by Words by Words By cluster to get Posts By cluster
PostsByClusters = PostsByWords.dot(WordsByCluster)

PostsByClusters = PostsByClusters.todense() * 1.0

row_min = PostsByFeatures.min(axis = 1)
row_max = PostsByFeatures.max(axis = 1)
row_diff_normed = (row_max - row_min == 0) + (row_max - row_min)
PostsByFeaturesNormed = (PostsByFeatures - row_min) / row_diff_normed

row_min = PostsByClusters.min(axis = 1)
row_max = PostsByClusters.max(axis = 1)
row_diff_normed = (row_max - row_min == 0) + (row_max - row_min)
PostsByClustersNormed = (PostsByClusters - row_min) / row_diff_normed

a = np.array(PostsByClusters)

len(a[0])

posts_df = pd.DataFrame(a)

rows, columns = posts_df.shape

import scipy
correlation_table =[]
for i in range(columns): # rows are the number of rows in the matrix. 
    correlation_row = []
    for j in range(columns):
        r = scipy.stats.pearsonr(a[:,i], a[:,j])
        correlation_row.append(r[0])
    correlation_table.append(correlation_row)

scipy.stats.pearsonr(a[:,19], a[:,18])

len(a[:,19])

# Print Correlation table
import csv
header = ["Cluster "+ str(i) for i in range(1,columns+1)]
with open('cluster-analysis/' + "correlation-"+model_name + "-" + str(num_word_clusters) + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([""]+header)
    for i in range(len(correlation_table)):
        writer.writerow([header[i]]+correlation_table[i])

num_posts_clusters =10
matricies = [PostsByFeatures, PostsByClusters, PostsByFeaturesNormed, PostsByClustersNormed]
names     = ["byFeatures", "byClusters", "byFeatures-Normed", "byClusters-Normed"]
mat_names = list(zip(matricies, names))
post_dfs  = []

for mat,name in mat_names:
    #initialize kmeans model
    kmeans = KMeans(n_clusters = num_posts_clusters, random_state = 42).fit(mat)
    # Save the clusters directory
    save_object(kmeans, 'clusters/', model_name + "-posts-" + name + "-" + str(num_posts_clusters))
    del kmeans

# Setup the header for the CSV files
header = ['total_posts', 'score_mean', 'score_median', 'score_range', 'comments_mean', 'comments_median', 'comments_range']
# Loop over all matricies
for mat,name in mat_names:
    # Load Clusters
    kmeans= load_object('clusters/', model_name + "-posts-" + name + "-" + str(num_posts_clusters))
    # Generate Post_clusters
    post_clusters = make_post_clusters(kmeans,mat,scores,num_comments_list)
    temp_header =header+list(map(lambda x:"element "+str(x),range(1,mat.shape[1]+1)))
    temp_table = list(map(lambda x: list(map(lambda y: x[1][y],header))+
                          list(map(lambda z: z[0],post_clusters[x[0]]['center'])),enumerate(post_clusters)))
    #post_dfs.append(pd.DataFrame.from_records(temp_table,columns =temp_header))

    import csv
    with open('post-analysis/' + model_name + '-' + str(num_posts_clusters) + '-' + name + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(temp_header)
        [writer.writerow(r) for r in temp_table]

