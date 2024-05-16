# for DSX, need to switch to the right directory. Detect using path name.
s = get_ipython().magic('pwd')
if s.startswith('/gpfs'):
    get_ipython().magic('cd ~/deep-learning-workshop/')

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# for making plots prettier
import seaborn as sns 
sns.set_style('white')

from __future__ import print_function
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

import notmnist
notmnist_path = "~/data/notmnist/notMNIST.pickle"

from display import visualize_keras_model, plot_training_curves
from helpers import combine_histories

# the data, shuffled and split between train, validation, and test sets
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = notmnist.load_data(notmnist_path)

len(x_train), len(x_valid), len(x_test)

# expect images...
plt.imshow(x_train[0])

# Confirm that labels are in order, with 'a' == 0
y_train[0], ord('g')-ord('a')

# Look at a bunch of examples
fig, axs = plt.subplots(20,20, sharex=True, sharey=True, figsize=(10,10))
for i, idx in enumerate(np.random.choice(len(x_train), 400, replace=False)):
    img = x_train[idx]
    ax = axs[i//20, i%20]
    ax.imshow(img)
    ax.axis('off')
sns.despine(fig, left=True, bottom=True)

# look at the distribution of values being used
fig, ax = plt.subplots(figsize=(3,2))
ax.hist(x_train[0].flatten(), bins=20);
sns.despine(fig)

x_train = x_train.reshape(-1, 784)
x_valid = x_valid.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
nb_classes = 10
nb_epoch = 10

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_valid = np_utils.to_categorical(y_valid, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_shape=(784,), name="hidden"))
model.add(Activation('relu', name="ReLU"))
model.add(Dense(10, name="output"))
model.add(Activation('softmax', name="softmax"))

model.summary()

# for multi-class classification, we'll use cross-entropy as the loss.
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              # Here, we tell Keras that we care about accuracy in addition to loss
              metrics=['accuracy'])

visualize_keras_model(model)

history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_training_curves(history.history);

history2 = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(x_valid, y_valid))

plot_training_curves(combine_histories(history.history, history2.history));

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model2 = Sequential()
model2.add(Dense(512, input_shape=(784,), name="hidden1"))
model2.add(Activation('relu', name="ReLU1"))
model2.add(Dropout(0.2))
model2.add(Dense(512, input_shape=(784,), name="hidden2"))
model2.add(Activation('relu', name="ReLU2"))
model2.add(Dropout(0.2))
model2.add(Dense(10, name="output"))
model2.add(Activation('softmax', name="softmax"))

model2.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

visualize_keras_model(model2)

# let's train for 20 epochs right away
history = model2.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid))

plot_training_curves(history.history);

model3 = Sequential()
model3.add(Dense(512, input_shape=(784,), name="hidden1"))
model3.add(Activation('relu', name="ReLU1"))
model3.add(Dropout(0.5))
model3.add(Dense(512, input_shape=(784,), name="hidden2"))
model3.add(Activation('relu', name="ReLU2"))
model3.add(Dropout(0.5))
model3.add(Dense(10, name="output"))
model3.add(Activation('softmax', name="softmax"))

model3.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model3.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid))

plot_training_curves(history.history);

history = model3.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid))

plot_training_curves(history.history);

score = model3.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

