get_ipython().magic('matplotlib inline')

from __future__ import print_function

from keras.datasets import imdb

import cPickle as pickle
import os.path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

max_features = 20000

# To work around bug in latest version of the dataset in Keras,
# load older version manually, downloaded from 
# https://s3.amazonaws.com/text-datasets/imdb_full.pkl
print('Loading data...')
path = os.path.expanduser('~/.keras/datasets/imdb_full.pkl')
f = open(path, 'rb')
(x_train, y_train), (x_test, y_test) = pickle.load(f)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# what does the data look like?
x_train[0]

word2index = imdb.get_word_index()

word2index.items()[:5]

# we want the other direction
index2word = dict([(i,w) for (w,i) in word2index.items()])

print('\n'.join([index2word[i] for i in range(1,20)]))

def totext(review):
    return ' '.join(index2word[i] for i in review)

for review in x_train[:10]:
    # let's look at the first 30 words
    print(totext(review[:30]))
    print('\n')

# what about labels?
y_train[1]

# how many labels?
np.unique(y_train)

# label balance
np.unique(y_train, return_counts=True)

lengths = map(len, x_train)
fig, axs = plt.subplots(2,1, figsize=(3,5))
axs[0].hist(lengths, bins=30)
axs[1].hist(lengths, bins=30, cumulative=True, normed=True)
axs[1].set_xlim([0,800])
axs[1].set_xticks(range(0,800,150))

sns.despine(fig)

