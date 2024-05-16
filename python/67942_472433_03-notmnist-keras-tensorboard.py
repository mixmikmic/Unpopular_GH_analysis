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

batch_size = 128
nb_classes = 10
nb_epoch = 10

# the data, shuffled and split between train, validation, and test sets
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = notmnist.load_data(notmnist_path)

x_train = x_train.reshape(-1, 784)
x_valid = x_valid.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_valid = np_utils.to_categorical(y_valid, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_shape=(784,), name="hidden"))
model.add(Activation('relu', name="ReLU"))
model.add(Dense(10, name="output"))
model.add(Activation('softmax', name="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.summary()

visualize_keras_model(model)

from keras.callbacks import TensorBoard

def make_tb_callback(run):
    """
    Make a callback function to be called during training.
    
    Args:
        run: folder name to save log in. 
    
    Made this a function since we need to recreate it when
    resetting the session. 
    (See https://github.com/fchollet/keras/issues/4499)
    """
    return TensorBoard(
            # where to save log file
            log_dir='./graph-tb-demo/' + run,
            # how often (in epochs) to compute activation histograms
            # (more frequently slows down training)
            histogram_freq=1, 
            # whether to visualize the network graph.
            # This now works reasonably in Keras 2.01!
            write_graph=True,
            # if true, write layer weights as images
            write_images=False)

tb_callback = make_tb_callback('1')

# and add it to our model.fit call
history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid),
                    callbacks=[tb_callback])

# Let's look at our manually visualized learning curves first
plot_training_curves(history.history);

# Uncomment if getting a "Invalid argument: You must feed a value
# for placeholder tensor ..." when rerunning training. 
# https://github.com/fchollet/keras/issues/4499
from keras.layers.core import K
K.clear_session() 
### 

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

tb_callback = make_tb_callback('complex')
history = model3.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=20,
                    verbose=1, validation_data=(x_valid, y_valid),
                     # Don't forget to include the callback!
                    callbacks=[tb_callback])

plot_training_curves(history.history);

