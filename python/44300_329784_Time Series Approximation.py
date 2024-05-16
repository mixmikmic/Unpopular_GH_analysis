import numpy as np
import matplotlib.pyplot as plt
# import math
# from sklearn.metrics import mean_squared_error

# function to be modelled
f = np.sin

number_of_period = 10
number_of_points_per_period = 20

x = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, number_of_points_per_period)
X = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)
plt.plot(X, np.sin(X), 'b')
plt.plot(x, np.sin(x), 'ro')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()

np.random.seed(42)
dataset = np.sin(X)

# train-test split
train_size = int(0.8*len(dataset))
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

len(train)

# convert a time series array into matrix of previous values
def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        data_x.append(a)
        data_y.append(dataset[i + look_back])
    return np.array(data_x), np.array(data_y)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

train[0], train[1], train[2]

trainX[0], trainX[1], trainX[2]

trainY[0], trainY[1], trainY[2]

trainX.shape

# reshape input
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

from keras.layers import Dense, LSTM
from keras.models import Sequential

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

len(trainPredict)

x = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)[:len(trainPredict)]
X = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)
plt.plot(X, np.sin(X), 'b')
plt.plot(x, trainPredict, 'r')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()

look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))

x = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)[:len(trainPredict)]
X = np.linspace(-number_of_period*np.pi, number_of_period*np.pi, 1000)
plt.plot(X, np.sin(X), 'b')
plt.plot(x, trainPredict, 'r')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()

f = np.exp

f(100)

number_of_points = 20

plt.close()
x = np.linspace(-10, 20, number_of_points_per_period)
X = np.linspace(-10, 20, 1000)
plt.plot(X, f(X), 'b')
plt.plot(x, f(x), 'r')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()

dataset = f(X)
# train-test split
train_size = int(0.8*len(dataset))
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

len(dataset)

look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))

from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(128, input_dim=look_back))
model.add(Dense(1))
adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))

from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(128, input_dim=look_back))
model.add(Dense(1))
adam = Adam(lr=100, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))

from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(128, input_dim=look_back))
model.add(Dense(1))
adam = Adam(lr=10000, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))

from keras.optimizers import Adam
model = Sequential()
model.add(LSTM(128, input_dim=look_back))
model.add(Dense(1))
adam = Adam(lr=1000000, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(trainX, trainY, nb_epoch=5, batch_size=1, validation_data=(testX, testY))



