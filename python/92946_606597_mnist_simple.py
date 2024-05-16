get_ipython().magic('matplotlib inline')
# import seaborn as sns

from keras.datasets import mnist
from keras import utils
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers import Conv2D, MaxPool2D

from keras_sequential_ascii import sequential_model_to_ascii_printout

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# (number of examples, x, y)
X_train.shape

X_test.shape

# 3 -> [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
Y_train = utils.to_categorical(y_train)
Y_test = utils.to_categorical(y_test)

# we need to add channel dimension (for convolutions)

# TensorFlow backend
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

# for Theano backend, it would be:
# X_train = X_train.reshape(-1, 1, 28, 28).astype('float32') / 255.
# X_test = X_test.reshape(-1, 1, 28, 28).astype('float32') / 255.

model = Sequential()

model.add(Flatten(input_shape=(28, 28, 1)))  # for Theano: (1, 28, 28)
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

sequential_model_to_ascii_printout(model)

# look at validation scores
model.fit(X_train, Y_train,
          epochs=10,
          validation_data=(X_test, Y_test))

model = Sequential()

model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

sequential_model_to_ascii_printout(model)

model.fit(X_train, Y_train,
          epochs=10,
          validation_data=(X_test, Y_test))

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

sequential_model_to_ascii_printout(model)

model.fit(X_train, Y_train,
          epochs=10,
          validation_data=(X_test, Y_test))

