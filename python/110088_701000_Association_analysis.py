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

PostsByClusters



sorted_clusters = sorted(list(zip(clusters,range(len(clusters)))),key = (lambda x : x[0]['total_freq']))

large_indicies = list(map(lambda x: x[1],sorted_clusters[-20:]))

sorted_large_indicies = sorted(large_indicies, reverse =True)

X = np.array(PostsByClusters.todense())
index_mapping = list(range(100))

for index in sorted_large_indicies:
    X = np.delete(X,index,1)
    del index_mapping[index]

assoc_confidence = 50
itemset_support  = 10

X_test = X[:700]

X_test

len(X_test)

itemsets = dict(frequent_itemsets(X_test > 0, itemset_support/
100))
assoc_rules = association_rules(itemsets, assoc_confidence/100)
rules = [(P, Q, supp, conf, conf/(itemsets[P]/X_test.shape[0]))
             for P, Q, supp, conf in assoc_rules
             if len(Q) == 1 and len(P)==1]

rules

rules    = load_object('association_rules/',model_name+'-assoc_rules-'+str(itemset_support)+
                       '-'+str(assoc_confidence)+'-'+str(num_word_clusters))
itemsets = load_object('itemsets/',model_name+'-itemset-'+str(itemset_support)+'-'+str(num_word_clusters))

len(rules)

len(itemsets)

len(rules)/len(itemsets)

rule_clusters =[]
for i in range(num_word_clusters):
    for lhs, rhs, support, confidence,lift in rules:
        if (i in lhs) or (i in rhs): 
            rule_clusters.append(i)
            break

len(rule_clusters)

rules.sort(key = lambda x : x[4],reverse = True)

filtered_rules = list(filter(lambda x: len(x[0])==1 and len(x[1])==1,rules ))

# load the models
model = gensim.models.Word2Vec.load('models/' + model_name + '.model')
kmeans = load_object('clusters/', model_name + "-words-cluster_model-" + str(num_word_clusters))
WordsByFeatures = load_object('matricies/', model_name + '-' + 'WordsByFeatures')

vocab_list = sorted(list(model.wv.vocab))

clusters = make_clustering_objects(model, kmeans, vocab_list, WordsByFeatures)

# Sort all the words in the words list
for cluster in clusters:
    cluster["word_list"].sort(key = lambda x:x[1], reverse = True)

len(filtered_rules)

import csv
top_num = min(10000,len(filtered_rules))
header = ["lhs","rhs","support","confidence","lift"]
with open('association-analysis/'+ model_name + "-filtered-lift-supp"+str(itemset_support) +
          "-conf-"+str(assoc_confidence)+'-'+ str(top_num) + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for i in range(top_num):
        rule = filtered_rules[i]
        lhs_top = clusters[index_mapping[next(iter(rule[0]))]]["word_list"][:5]
        rhs_top = clusters[index_mapping[next(iter(rule[1]))]]["word_list"][:5]
        writer.writerow([lhs_top,rhs_top ,rule[2],rule[3],rule[4]])



