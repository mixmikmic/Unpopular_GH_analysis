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
model_name = "example_model"

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

PostsByClusters

itemsets = dict(frequent_itemsets(PostsByClusters > 0, .40))

assoc_rules = association_rules(itemsets,0.8)

rules = [(P, Q, supp, conf, conf/(itemsets[P]/PostsByClusters.shape[0]))
         for P, Q, supp, conf in association_rules(itemsets, .95)]

for lhs, rhs, support, confidence,lift in rules:
    print(", ".join([str(i) for i in lhs]), "-->",", ".join([str(i) for i in rhs]), "support: ",
          support, " confidence: ",confidence, "lift: ", lift)

len(rules)

rule_clusters =[]
for i in range(100):
    for lhs, rhs, support, confidence,lift in rules:
        if (i in lhs) or (i in rhs): 
            rule_clusters.append(i)
            break

rule_clusters

len(rule_clusters)

save_object(rules,'objects/',model_name+'-assoc_rules')
save_object(itemsets,'objects/',model_name+'-itemset')

