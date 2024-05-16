from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_train, height, width = X_train.shape
n_test, _, _ = X_test.shape
print("Number of samples in training data = {}.\nDimensions of each sample are {}X{}".format(n_train, height, width))
print("Number of samples in test data = {}".format(n_test))

print(X_train.shape)
print(X_train[0].shape)

import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.show()

print(y_train)

from keras.utils.np_utils import to_categorical
from keras import backend as K

# Reshaping data

# this if condition is needed because the input shape is specified differently for theano and tensorflow backend.
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(n_train, 1, height, width).astype('float32')
    X_test = X_test.reshape(n_test, 1, height, width).astype('float32')
    input_shape = (1, height, width)
else:
    X_train = X_train.reshape(n_train, height, width, 1).astype('float32')
    X_test = X_test.reshape(n_test, height, width, 1).astype('float32')
    input_shape = (height, width, 1)
    

# Normalizing data
X_train /= 255
X_test /= 255

# Transforming output variables
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

print(X_train.shape)
print(X_train[0].shape)

print(y_train)

print(y_train[0])
print(y_train[1])

from keras.models import Sequential
model = Sequential()

from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

# Convolution Layer

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5, activation='relu'))

# Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Regularization Layer
model.add(Dropout(0.25))

# Flatten Layer
model.add(Flatten())

# Fully Connected
model.add(Dense(128))
model.add(Activation('relu'))

# Regularization Layer
model.add(Dropout(0.5))

# Fully Connected with softmax
model.add(Dense(n_classes))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X_train, y_train, batch_size=128, nb_epoch=2, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)



