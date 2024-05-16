# We can use the `time` and the `numpy` module to time how long it takes to do an FFT
from time import time
import numpy as np
import seaborn as sns
sns.set_style('white')

# For a signal of length ~1000. Say, 100ms of a 10KHz audio sample.
signal = np.random.randn(1009)
start = time()
_ = np.fft.fft(signal)
stop = time()
print('It takes {} seconds to do the FFT'.format(stop-start))

# We'll test out how long the FFT takes for a few lengths
test_primes = [11, 101, 1009, 10009, 100019]

# Let's try a few slightly longer signals
for i_length in test_primes:
    # Calculate the number of factors for this length (we'll see why later)
    factors = [ii for ii in range(1, 1000) if i_length % ii == 0]
    # Generate a random signal w/ this length
    signal = np.random.randn(i_length)
    # Now time the FFT
    start = time()
    _ = np.fft.fft(signal)
    stop = time()
    print('With data of length {} ({} factors), it takes {} seconds to do the FFT'.format(
            i_length, len(factors), stop-start))

import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

# Let's read in the data and see how it looks
df = pd.read_csv('../data/fft_times.csv', index_col=0)
df = df.apply(pd.to_numeric)
df = df.mean(0).to_frame('time')
df.index = df.index.astype(int)
df['length'] = df.index.values

# First off, it's clear that computation time grows nonlinearly with signal length
df.plot('length', 'time', figsize=(10, 5))

# However, upon closer inspection, it's clear that there's much variability
winsize = 500
i = 0
j = i + winsize
df.iloc[i:j]['time'].plot(figsize=(10, 5))

# We'll use a regression model to try and fit how length predicts time
mod = linear_model.LinearRegression()
xfit = df['length']
xfit = np.vstack([xfit, xfit**2, xfit**3, xfit**4]).T
yfit = df['time'].reshape([-1, 1])

# Now fit to our data, and calculate the error for each datapoint
mod.fit(xfit, yfit)
df['ypred'] = mod.predict(xfit)
df['diff'] = df['time'] - df.ypred

# As the length grows, the trends in the data begin to diverge more and more
ax = df.plot('length', 'diff', kind='scatter',
             style='.', alpha=.5, figsize=(10, 5))
ax.set_ylim([0, .05])
ax.set_title('Error of linear fit for varying signal lengths')

# We'll write a helper function that shows how many (<100) factors each length has
find_n_factors = lambda n: len([i for i in range(1, min(100, n-1)) if n % i == 0])

# This tells us the number of factors for all lengths we tried
df['n_factors'] = df['length'].map(find_n_factors)

# We now have a column that tells us how many factors each iteration had
df.tail()

# As we can see, the FFT time drops quickly as a function of the number of factors
ax = df.plot('n_factors', 'time', style=['.'], figsize=(10, 5), alpha=.1)
ax.set_xlim([0, 15])
ax.set_ylabel('Time for FFT (s)')
ax.set_title('Time of FFT for varying numbers of factors')

# We'll plot two zoom levels to see the detail
f, axs = plt.subplots(2, 1, figsize=(10, 5))
vmin, vmax = 1, 18
for ax in axs:
    ax = df.plot.scatter('length', 'time', c='n_factors', lw=0, cmap=plt.cm.get_cmap('RdYlBu', vmax),
                                           figsize=(10, 10), vmin=vmin, vmax=vmax, ax=ax, alpha=.5)
    ax.set_xlabel('Length of signal (samples)')
    ax.set_ylabel('Time to complete FFT (s)')
    ax.set_title('Time to compute the FFT, colored by n_factors')
_ = plt.setp(axs, xlim=[0, df['length'].max()])
_ = plt.setp(axs[0], ylim=[0, .2])
_ = plt.setp(axs[1], ylim=[0, .005])
plt.tight_layout()

