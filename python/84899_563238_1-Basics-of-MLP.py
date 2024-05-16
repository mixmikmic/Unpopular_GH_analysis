from keras.datasets import boston_housing

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.models import Sequential

model = Sequential()

from keras.layers import Activation, Dense

# Keras model with two hidden layer with 10 neurons each 
model.add(Dense(10, input_shape = (13,)))    # Input layer => input_shape should be explicitly designated
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # Hidden layer => only output dimension should be designated
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # Hidden layer => only output dimension should be designated
model.add(Activation('sigmoid'))
model.add(Dense(1))                          # Output layer => output dimension = 1 since it is regression problem

# This is equivalent to the above code block
model.add(Dense(10, input_shape = (13,), activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1))

from keras import optimizers

sgd = optimizers.SGD(lr = 0.01)    # stochastic gradient descent optimizer

model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse'])    # for regression problems, mean squared error (MSE) is often employed

model.summary()

model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 1)

results = model.evaluate(X_test, y_test)

print(model.metrics_names)     # list of metric names the model is employing
print(results)                 # actual figure of metrics computed

print('loss: ', results[0])
print('mse: ', results[1])

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()

X_data = whole_data.data
y_data = whole_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 7) 

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.models import Sequential

model = Sequential()

# Keras model with two hidden layer with 10 neurons each 
model.add(Dense(10, input_shape = (30,)))    # Input layer => input_shape should be explicitly designated
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # Hidden layer => only output dimension should be designated
model.add(Activation('sigmoid'))
model.add(Dense(10))                         # Hidden layer => only output dimension should be designated
model.add(Activation('sigmoid'))
model.add(Dense(1))                          # Output layer => output dimension = 1 since it is regression problem
model.add(Activation('sigmoid'))

# This is equivalent to the above code block
model.add(Dense(10, input_shape = (13,), activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

from keras import optimizers

sgd = optimizers.SGD(lr = 0.01)    # stochastic gradient descent optimizer

model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 1)

results = model.evaluate(X_test, y_test)

print(model.metrics_names)     # list of metric names the model is employing
print(results)                 # actual figure of metrics computed

print('loss: ', results[0])
print('accuracy: ', results[1])

