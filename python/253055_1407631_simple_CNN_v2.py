import os
import sys
import pandas as pd
import numpy as np

seed = 16
np.random.seed(seed)

from keras.utils.np_utils import to_categorical

from tensorflow.python.client import device_lib
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
print(device_lib.list_local_devices())

from scipy.io import loadmat
os.chdir('C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets')

# reading in the test/train data for images and labels.

train_data = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_data.mat''')['train_data']
train_labels = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_list.mat''')['labels']
test_data = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\test_data.mat''')['test_data']
test_labels = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\test_list.mat''')['labels']

from sklearn.model_selection import train_test_split

print(train_data.shape)
print(train_labels.shape)

df = pd.DataFrame(train_data)
df.head()

labels = [item for label in train_labels for item in label] 
df2 = pd.DataFrame({'label':labels})

pre_split = pd.concat([df, df2], axis=1)
pre_split.head()

train, validate = train_test_split(df, test_size = 0.2, stratify=train_labels, random_state=16)

train.head()

#might have finally figured out how to stratify the image data...

X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, stratify=train_labels, random_state=16)

# transform the y_train and y_val to categoricals  ***not sure if this is needed**
y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)
num_classes = y_val.shape[0]
num_classes

X_train.shape

# using a simple CNN to start

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('tf')

input_shape = (9600, 12000, 0)

# create the model

model = Sequential()
model.add(Conv2D(64, (11, 11), strides=4, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(32, (11, 11), strides=4, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(16, (11, 11), strides=4, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
     

# Compile model
epochs = 25
lrate = 0.003
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())



# Fit the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32)
scores = model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

