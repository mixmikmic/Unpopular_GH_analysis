import pandas as pd
data = pd.read_csv('../data/bikes.csv', sep=',')
data.head()

data.sort_values(['dteday', 'hr'])
cnt = data['cnt']
data['hist'] = cnt.shift(1)
data = data[1:]
data.head()

from sklearn.model_selection import train_test_split

X_all = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'atemp', 'hum', 'windspeed', 'hist']]
y_all = data['cnt']


X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all,
    random_state=1,
    test_size=0.2)

print('Train size: {}'.format(len(X_train)))
print('Test size: {}'.format(len(X_test)))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential()

model.add(Dense(32, input_shape=(13, )))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])

model.fit(X_train, y_train,
          batch_size = 128, epochs = 500, verbose=1,
          validation_data=(X_test, y_test))

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_pred = model.predict(X_test)
print ("Test mean: {}, std: {}".format(np.mean(y_test), np.std(y_test)))
print("Test Root mean squared error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
print("Test Mean absolute error: {:.2f}".format(mean_absolute_error(y_test, y_pred)))

y_pred = model.predict(X_train)
print("Train Root mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_train, y_pred)))
print("Train Mean absolute error: %.2f"
      % mean_absolute_error(y_train, y_pred))

