from __future__ import division, print_function, absolute_import
import re, collections
import numpy as np
import pandas as pd

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_2d_transpose
from tflearn.layers.estimator import regression

from IPython import display
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
pd.set_option('max_rows', 100)
# np.set_printoptions(threshold=np.nan)

df = pd.read_csv('modified.csv',parse_dates = ['date'], 
                 infer_datetime_format = True)

df.drop(['date','b0','a0','fut_direction'], 1, inplace = True)
df.sort_index(axis=1, inplace = True)
df

# Bringing the data in the shape as shown in 'How data looks.xlsx excel file'
X = df.values

print('Original X')
print(X.shape)

seq_length = 300
img_height = 20
num_images = X.shape[0]//seq_length

X = X[:seq_length*num_images,:]
print('\nX after calculating num_images')
print(X.shape)

X = X.reshape(num_images,seq_length, img_height).astype("float32").transpose((0,2,1))

Y = np.array([X[i+1,:,0] for i,x in enumerate(X) if i<X.shape[0]-1]).astype("int64")
print('\nChecking Y \nFirst time series value of X[1]')
print(X[1,:,0])
print('Value of Y corresponding to X[0]')
print(Y[0])
print('\nX and Y after converting to 20x300 time-series images')
X = X[:-1,:,:]
print(X.shape)
print(Y.shape)

train_split = 0.85
split_val = round(num_images*train_split)

X_train = X[:split_val,:,:,np.newaxis]
X_test = X[split_val:,:,:,np.newaxis]
Y_train = Y[:split_val,:]
Y_test = Y[split_val:,:]

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_train, axis=0)
X_test /= np.std(X_train, axis=0)

print('\nX and Y after train-test split, normalisation and channel dimension insertion')
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

tf.reset_default_graph()
filter_size = 3
stride_y = 5
epochs = 50

# Convolutional network with 2 conv and 2 fully connected layers
network = input_data(shape=[None, 20, 300, 1])
conv1 = conv_2d(network, 32, filter_size, activation='relu', padding = 'same', strides= [1,1,stride_y,1], name = 'conv1')
pool1 = max_pool_2d(conv1, 2)
conv2 = conv_2d(pool1, 64, filter_size, activation='relu', padding = 'same', name ='conv2')
pool2 = max_pool_2d(conv2, 2)
fc1 = fully_connected(pool2, 512, activation='relu',name ='fc1')
drop1 = dropout(fc1, 0.5)
fc2 = fully_connected(drop1, 20, name ='fc2')
network = regression(fc2, optimizer='adam', loss='mean_square',
                     learning_rate=0.001, metric='R2')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X_train, Y_train, n_epoch=epochs, shuffle=False, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=26,run_id='irage')

conv1 = model.get_weights(conv1.W)
conv2 = model.get_weights(conv2.W)
fc1 = model.get_weights(fc1.W)
fc2 = model.get_weights(fc2.W)

tf.trainable_variables()

from PIL import Image

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[256, 256] = [255, 0, 0]
img = Image.fromarray(data, 'RGB')
img.save('my.png')

t = np.mean(conv2, axis=3, keepdims = True)
t = np.mean(t, axis=2, keepdims = True)

conv2_avg = tf.constant(t)
conv2_decon = conv_2d_transpose(conv2_avg, 1, filter_size, output_shape = [2, 2, 1])
conv1_avg = tf.constant(np.mean(conv1, axis=3))
conv1_conv2 = tf.multiply(conv1_avg,conv2_decon)
mask = conv_2d_transpose(conv1_conv2, 1, filter_size, [20,300,1])

sess = tf.Session()
init = tf.global_variables_initializer()
result = sess.run(mask)
print(result)



