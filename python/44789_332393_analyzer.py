from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

help(tflearn.input_data)

#IMDB Dataset loading
#load_data in module tflearn.datasets.imdb:
#extension is picke which means a bit steam, so it make it easier to convert to other 
#python objects like list, tuple later 
#param n_words: The number of word to keep in the vocabulary.All extra words are set to unknow (1).
#param valid_portion: The proportion of the full train set used for the validation set.

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
#function pad_sequences in module tflearn.data_utils:
# padding is necessary to have consistency in our input dimensionality
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

#Now we can convert output to binaries where 0 means Negative and 1 means Positive
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

print(trainX[1:10,:])
print(trainX[0,:].shape)
type(trainX[0,0])

# Network building
#function input_data in module tflearn.layers.core:
#`input_data` is used as a data entry (placeholder) of a network.
#This placeholder will be feeded with data when training
# shape: list of `int`. An array or tuple representing input data shape.
#            It is required if no placeholder provided. First element should
#            be 'None' (representing batch size), if not provided, it will be
#            added automatically.
net = tflearn.input_data([None, 100])

# LAYER-1
# input_dim = 10000, since that how many words we loaded from our dataset
# output_dim = 128, number of outputs of resulting embedding

# LAYER-2
net = tflearn.embedding(net, input_dim=10000, output_dim=128)

# LSTM = long short term memory
# this layer allow us to remember our data from the beginning of the sequences which 
# allow us to improve our prediction
# dropout = 0.8, is the technique to prevent overfitting, which randomly switch on and off different 
# pathways in our network
net = tflearn.lstm(net, 128, dropout=0.8)

# LAYER-3: Our next layer is fully connected, this means every neuron in the previous layer is connected to 
# the every neuron in this layer
# Good way of learning non-linear combination
# activation: take in vector of input and squash it into vector of output probability between 
# 0 and 1.
net = tflearn.fully_connected(net, 2, activation='softmax')

# LAYER-4: 
# adam : perform gradient descent
# categorical_crossentropy: find error between predicted output and original output 
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

# Training
# Initialize with tflearn Deep Neural Network Model.
# tensorboard_verbose: `int`. Summary verbose level, it accepts
#          different levels of tensorboard logs:
#          0: Loss, Accuracy (Best Speed).
#          1: Loss, Accuracy, Gradients.
#          2: Loss, Accuracy, Gradients, Weights.
#          3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
#              (Best visualization)
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,batch_size=32)

help(tflearn.DNN)

testY

predictionsY = model.predict(testX)

import matplotlib.pyplot as plt

plt.scatter(testX[:,1], testX[:,2])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Imdb Film Dataset')
plt.show()

