import math, time
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import numpy as np

import keras
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

get_ipython().run_cell_magic('javascript', '', "//Creating shortcut 'r' to run all the below cells\nJupyter.keyboard_manager.command_shortcuts.add_shortcut('r', {\n    help : 'run below cells',\n    help_index : 'zz',\n    handler : function (event) {\n        IPython.notebook.execute_cells_below();\n        return false;\n    }}\n);\n\n//Creating shortcut 'l' to run all the cells\nJupyter.keyboard_manager.command_shortcuts.add_shortcut('l', {\n    help : 'run all cells',\n    help_index : 'zz',\n    handler : function (event) {\n        IPython.notebook.execute_all_cells();\n        return false;\n    }}\n);")

stock_name = 'AMZN'
start = dt.datetime(1995,1,1)
end   = dt.date.today()
df = web.DataReader(stock_name, 'google', start, end)
df.to_csv('%s_data.csv'%stock_name, header=True, index=False)
df = pd.read_csv('%s_data.csv'%stock_name)

# Dropping all columns except 'Open','High' and 'Close'
df.drop(['Low','Volume'], axis = 1, inplace=True)
df.head()

#Method1 - Division by 10
df = df/(10^(len(str(df.iloc[0,0]).split('.')[0])-1))
#Method2 - Division by 1st value
# df = df/df.iloc[0,0]
df.head()

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = int(round(0.9 * result.shape[0]))
    train = result[:row, :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    d = 0.2
    init = glorot_uniform(seed = 69)

    model = Sequential()
    model.add(LSTM(32, input_shape=(layers[0], layers[1]), return_sequences=True, kernel_initializer = init))
    model.add(Dropout(d))
    model.add(LSTM(32, return_sequences=False, kernel_initializer = init))
    model.add(Dropout(d))
    model.add(Dense(8,kernel_initializer=init ,activation='relu'))        
    model.add(Dense(1,kernel_initializer= init ,activation='linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model

window = 22
X_train, y_train, X_test, y_test = load_data(df[::-1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

model = build_model([window, 3])

stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta= 0.5 , patience=3, verbose=2, mode='auto')
start_time = time.time()

model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
#     validation_split=0.1,
#     callbacks = [stop],
    verbose = 2)

print('\nTime taken for training: %.2f minutes'%((time.time()-start_time)/60))

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

# print(X_test[-1])
diff=[]
ratio=[]
p = model.predict(X_test)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))

plt.plot(p,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')
plt.legend(loc='upper left')
plt.show()

