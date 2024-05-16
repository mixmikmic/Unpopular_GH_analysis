get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

# try using other layers
from keras.layers import Conv2D, MaxPool2D, Dropout

# optionally
# install with: 
from keras_sequential_ascii import sequential_model_to_ascii_printout

# load data
data = io.loadmat("notMNIST_small.mat")

# transform data
X = data['images']
y = data['labels']
resolution = 28
classes = 10

X = np.transpose(X, (2, 0, 1))

y = y.astype('int32')
X = X.astype('float32') / 255.

# channel for X
X = X.reshape((-1, resolution, resolution, 1))

# 3 -> [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
Y = np_utils.to_categorical(y, 10)

# looking at data; some fonts are strange
i = 42
print("It is:", "ABCDEFGHIJ"[y[i]])
plt.imshow(X[i,:,:,0]);

# splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# creating a simple netural network
# in this case - just logistic regression
model = Sequential()

# add Conv2D and MaxPool2D layers

model.add(Flatten(input_shape=(resolution, resolution, 1)))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

sequential_model_to_ascii_printout(model)

model.fit(X_train, Y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, Y_test))



