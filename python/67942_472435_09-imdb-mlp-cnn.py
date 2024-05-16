from __future__ import print_function

import os
import os.path
import sys
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns

from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer

from display import visualize_keras_model, plot_training_curves

get_ipython().magic('matplotlib inline')
sns.set_style('white')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# To work around bug in latest version of the dataset in Keras,
# load older version manually, downloaded from 
# https://s3.amazonaws.com/text-datasets/imdb_full.pkl
print('Loading data...')
path = os.path.expanduser('~/.keras/datasets/imdb_full.pkl')
f = open(path, 'rb')
(x_train, y_train), (x_test, y_test) = pickle.load(f)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

word2index = imdb.get_word_index()

# we want the other direction
index2word = dict([(i,w) for (w,i) in word2index.items()])

def totext(review):
    return ' '.join(index2word[i] for i in review)

GLOVE_DIR = os.path.expanduser('~/data/glove')
MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {} # word -> coefs
# We'll use the 100-dimensional version
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# Shuffle training and test data
# (copying from imdb.py because we had to load data manually)
seed = 113
np.random.seed(seed)
np.random.shuffle(x_train)
np.random.seed(seed) # same shuffle for labels!
np.random.shuffle(y_train)

np.random.seed(seed * 2)
np.random.shuffle(x_test)
np.random.seed(seed * 2)
np.random.shuffle(y_test)

print('Pad sequences')
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Recall: words are indexed in descending order of frequency. Remove the 
# less frequent ones -- replace with constant value
x_train[x_train >= MAX_NB_WORDS] = MAX_NB_WORDS-1
x_test[x_test >= MAX_NB_WORDS] = MAX_NB_WORDS-1

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word2index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# train a regular MLP
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)
x = Flatten()(embedded_sequences)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model_mlp = Model(sequence_input, preds)
model_mlp.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model_mlp.summary()

print('Training model.')
history = model_mlp.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=25, batch_size=128)

plot_training_curves(history.history);

score, acc = model_mlp.evaluate(x_test, y_test,
                            batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(7)(x)  # modified from example since our seq len is 300 
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()

print('Training model.')
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=10, batch_size=128)

plot_training_curves(history.history);

score, acc = model.evaluate(x_test, y_test,
                            batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)

print('Build model...')
batch_size = 32
maxlen = 80

x_train_short = pad_sequences(x_train, maxlen=maxlen)
x_test_short = pad_sequences(x_train, maxlen=maxlen)

model_lstm = Sequential()
model_lstm.add(Embedding(MAX_NB_WORDS, 128))
model_lstm.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model_lstm.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model_lstm.fit(x_train_short, y_train, batch_size=batch_size, epochs=15,
          validation_data=(x_test_short, y_test))

score, acc = model_lstm.evaluate(x_test_short, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

