from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import collections
import random
from time import time

from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA

import data_handler as dh
import semeval_data_helper as sdh


# plot settings
get_ipython().magic('matplotlib inline')
# print(plt.rcParams.keys())
# plt.rcParams['figure.figsize'] = (16,9)

import mpld3

# reload(eh)
import experiment_helper as eh

shuffle_seed = 20

reload(dh)
DH = dh.DataHandler('data/semeval_train_sdp__include_8000', valid_percent=10, shuffle_seed=shuffle_seed) # for semeval

# reload(sdh)
train, valid, test, label2int, int2label = sdh.load_semeval_data(include_ends=True, shuffle_seed=shuffle_seed)
num_classes = len(int2label.keys())

### WE DONT WANT INDICES THIS TIME ###
# # convert the semeval data to indices under the wiki vocab:
# train['sdps'] = DH.sentences_to_sequences(train['sdps'])
# valid['sdps'] = DH.sentences_to_sequences(valid['sdps'])
# test['sdps'] = DH.sentences_to_sequences(test['sdps'])
    
# train['targets'] = DH.sentences_to_sequences(train['targets'])
# valid['targets'] = DH.sentences_to_sequences(valid['targets'])
# test['targets'] = DH.sentences_to_sequences(test['targets'])

# print(train['targets'][:5]) # small sample

# max_seq_len = max([len(path) for path in train['sdps']+valid['sdps']+test['sdps']])
# print(max_seq_len, DH.max_seq_len)
# DH.max_seq_len = max_seq_len

from sklearn.pipeline import Pipeline
# define baseline pipelines
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Feature Extractors
cv = CountVectorizer(
        input=u'content', 
        encoding=u'utf-8', 
        decode_error=u'strict', 
        strip_accents='unicode', 
        lowercase=True,
        analyzer=u'word', 
        preprocessor=None, 
        tokenizer=None, 
        stop_words='english', 
        #token_pattern=u'(?u)\\b\w\w+\b', # one alphanumeric is a token
        ngram_range=(1, 2), 
        max_df=.9, 
        min_df=2, 
        max_features=None, 
        vocabulary=None, 
        binary=False, 
        #dtype=type 'numpy.int64'>
        )
from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer(
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
)

# Final Classifier
lr = LogisticRegression(C=.05,
                        fit_intercept=True,
                        random_state=0,
                        class_weight='balanced',
#                         multi_class='multinomial',
                        #solver='lbfgs',
                        n_jobs=1)

pipeline = Pipeline([
    ('count', cv),
    ('tfidf', tf),
    ('logreg', lr)
    ])

param_grid = {
    'count__ngram_range':[(1,1),(1,2),(1,3)],
    'tfidf__norm':['l1', 'l2'],
    'tfidf__use_idf':[True, False],
    'tfidf__sublinear_tf':[True,False],
    'logreg__C':[.001, .01, .1],
    'logreg__penalty':['l1', 'l2']
}

from sklearn.grid_search import GridSearchCV
grid_search = GridSearchCV(pipeline, 
                           param_grid,
                           scoring='f1_macro',
                           n_jobs=-1, verbose=1)

print("Here")
x_data = [sent[0].text for sent in train['sents']]
y_data = train['labels']
print(x_data[0], y_data[0])
grid_search.fit(np.array(x_data), y_data)
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

from sklearn.metrics import f1_score
test_x = [sent[0].text for sent in valid['sents']]
test_y = valid['labels']

lr_sent = grid_search.best_estimator_
lr_sent.fit(x_data, y_data)
preds = lr_sent.predict(test_x)
lr_sent_score = f1_score(test_y, preds, average='macro')
print('LR Sentence validation f1 macro: %2.4f' % lr_sent_score)

from sklearn.pipeline import Pipeline
# define baseline pipelines
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Feature Extractors
cv = CountVectorizer(
        input=u'content', 
        encoding=u'utf-8', 
        decode_error=u'strict', 
        strip_accents='unicode', 
        lowercase=True,
        analyzer=u'word', 
        preprocessor=None, 
        tokenizer=None, 
        stop_words='english', 
        #token_pattern=u'(?u)\\b\w\w+\b', # one alphanumeric is a token
        ngram_range=(1, 2), 
        max_df=.9, 
        min_df=2, 
        max_features=None, 
        vocabulary=None, 
        binary=False, 
        #dtype=type 'numpy.int64'>
        )
from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer(
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
)

# Final Classifier
lr = LogisticRegression(C=.05,
                        fit_intercept=True,
                        random_state=0,
                        class_weight='balanced',
#                         multi_class='multinomial',
                        #solver='lbfgs',
                        n_jobs=1)

pipeline = Pipeline([
    ('count', cv),
    ('tfidf', tf),
    ('logreg', lr)
    ])

param_grid = {
    'count__ngram_range':[(1,1),(1,2),(1,3)],
    'tfidf__norm':['l1', 'l2'],
    'tfidf__use_idf':[True, False],
    'tfidf__sublinear_tf':[True,False],
    'logreg__C':[.001, .01, .1],
    'logreg__penalty':['l1', 'l2']
}

from sklearn.grid_search import GridSearchCV
grid_search = GridSearchCV(pipeline, 
                           param_grid,
                           scoring='f1_macro',
                           n_jobs=-1, verbose=1)

print("Here")
x_data = [" ".join([tok[0] for tok in sdp]) for sdp in train['sdps']]
y_data = train['labels']
print(x_data[0], y_data[0])
grid_search.fit(np.array(x_data), y_data)
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

lr_sdp = grid_search.best_estimator_
lr_sdp.fit(x_data, y_data)
preds = lr_sdp.predict(test_x)
lr_sdp_score = f1_score(test_y, preds, average='macro')
print('LR SDP validation f1 macro: %2.4f' % lr_sdp_score)

brown_clusters = {}
lines = open('data/shuffled.en.tok-c50-p1.out/paths', 'r').readlines()
for line in lines:
    vec = line.split()
    brown_clusters[vec[1]] = vec[0]
del lines

# append brown clusters to each sdp sentence
x_data = [" ".join([tok[0] for tok in sdp]) + " " + 
          " ".join([brown_clusters[str(tok[0])] for tok in sdp if str(tok[0]) in brown_clusters])
          for sdp in train['sdps']]
print(x_data[:5])
test_x = [" ".join([tok[0] for tok in sdp]) + " " + 
          " ".join([brown_clusters[str(tok[0])] for tok in sdp if str(tok[0]) in brown_clusters])
          for sdp in valid['sdps']]
x_data_ends = [" ".join([tok[0] for tok in sdp]) + " " + 
          " ".join([brown_clusters[str(tok[0])] for tok in [sdp[0], sdp[-1]] if str(tok[0]) in brown_clusters])
          for sdp in train['sdps']]
test_x_ends = [" ".join([tok[0] for tok in sdp]) + " " + 
          " ".join([brown_clusters[str(tok[0])] for tok in [sdp[0], sdp[-1]] if str(tok[0]) in brown_clusters])
          for sdp in valid['sdps']]

from sklearn.pipeline import Pipeline
# define baseline pipelines
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Feature Extractors
cv = CountVectorizer(
        input=u'content', 
        encoding=u'utf-8', 
        decode_error=u'strict', 
        strip_accents='unicode', 
        lowercase=True,
        analyzer=u'word', 
        preprocessor=None, 
        tokenizer=None, 
        stop_words='english', 
        #token_pattern=u'(?u)\\b\w\w+\b', # one alphanumeric is a token
        ngram_range=(1, 2), 
        max_df=.9, 
        min_df=2, 
        max_features=None, 
        vocabulary=None, 
        binary=False, 
        #dtype=type 'numpy.int64'>
        )
from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer(
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
)

# Final Classifier
lr = LogisticRegression(C=.05,
                        fit_intercept=True,
                        random_state=0,
                        class_weight='balanced',
#                         multi_class='multinomial',
                        #solver='lbfgs',
                        n_jobs=1)

pipeline = Pipeline([
    ('count', cv),
    ('tfidf', tf),
    ('logreg', lr)
    ])

param_grid = {
    'count__ngram_range':[(1,1),(1,2),(1,3)],
    'tfidf__norm':['l1', 'l2'],
    'tfidf__use_idf':[True, False],
    'tfidf__sublinear_tf':[True,False],
    'logreg__C':[.001, .01, .1, .5, 1.],
    'logreg__penalty':['l1', 'l2']
}

from sklearn.grid_search import GridSearchCV
grid_search = GridSearchCV(pipeline, 
                           param_grid,
                           scoring='f1_macro',
                           n_jobs=-1, verbose=1)

print(x_data[0], y_data[0])
grid_search.fit(np.array(x_data), y_data)
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

lr_sdp_brown = grid_search.best_estimator_
lr_sdp_brown.fit(x_data, y_data)
preds = lr_sdp_brown.predict(test_x)
lr_sdp_brown_score = f1_score(test_y, preds, average='macro')
print('LR SDP validation f1 macro: %2.4f' % lr_sdp_brown_score)

test_x = [" ".join([tok[0] for tok in sdp]) + " " + 
          " ".join([brown_clusters[str(tok[0])] for tok in sdp if str(tok[0]) in brown_clusters])
          for sdp in test['sdps']]

preds = lr_sdp_brown.predict(test_x)
with open('SemEval2010_task8_all_data/test_pred.txt', 'w') as f:
    i = 8001
    for pred in preds:
        f.write("%i\t%s\n" % (i, int2label[pred]))
        i += 1

get_ipython().run_cell_magic('bash', '', './SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl \\\nSemEval2010_task8_all_data/test_pred.txt SemEval2010_task8_all_data/test_keys.txt')



