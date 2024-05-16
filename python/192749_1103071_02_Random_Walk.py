import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# create and plot a random walk
from random import seed
from random import random

seed(1)
random_walk = list()

# 1.
random_walk.append(-1 if random() < 0.5 else 1)

# 2., 3.
for i in range(1, 1000):
    movement = -1 if random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)

plt.plot(random_walk);

# Autocorrelation
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(random_walk);

# prepare dataset
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]

# Persistence Prediction
predictions = list()
history = train[-1]
for i in range(len(test)):
    y_hat = history
    predictions.append(y_hat)
    history = test[i]

# Evaluation
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(test, predictions))
print('Persistence RMSE: %.3f' % rmse)

# prepare dataset
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]

# random prediction
predictions = list()
history = train[-1]
for i in range(len(test)):
    y_hat = history + (-1 if random() < 0.5 else 1)
    predictions.append(y_hat)
    history = test[i]

rmse = sqrt(mean_squared_error(test, predictions))
print('Persistence RMSE: %.3f' % rmse)

