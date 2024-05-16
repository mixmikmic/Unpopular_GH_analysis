import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

series = pd.read_csv(
    './data/daily-total-female-births.csv',
    header=0,
    index_col=0,
    parse_dates=True,
    squeeze=True)

series.head()

# tail-rolling average transform
rolling = series.rolling(window=3)
rolling_mean = rolling.mean()

rolling_mean.head(10)

# plot original and transformed dataset
series.plot()
rolling_mean.plot(color='red')

plt.show()

# zoomed plot original and transformed dataset
series[:100].plot()
rolling_mean[:100].plot(color='red')

plt.show()

df = pd.DataFrame(series.values)

width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)

# mean = mean(t-2, t-1, t)
means = window.mean()
df = pd.concat([means, lag1, df], axis=1)
df.columns = ['mean', 't', 't+1']

df.head(10)

# prepare situation
X = series.values

window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()

# walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    y_hat = np.mean([history[i] for i in range(length-window,length)])
    obs = test[t]
    
    predictions.append(y_hat)
    history.append(obs)
    print('predicted=%f, real_val=%f' % (y_hat, obs))

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot
plt.plot(test)
plt.plot(predictions, color='red')

plt.show()

# zoom plot
plt.plot(test[:100])
plt.plot(predictions[:100], color='red')

plt.show()

