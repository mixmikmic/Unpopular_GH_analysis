# important stuff:
import os
import pandas as pd
import numpy as np
import statsmodels.tools.numdiff as smnd
import scipy

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Magic function to make matplotlib inline;
# other style specs must come AFTER
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
# There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14

n = 50  # number of data points
x = np.linspace(-10, 10, n)
yerr = np.abs(np.random.normal(0, 2, n))
y = np.linspace(5, -5, n) + np.random.normal(0, yerr, n)
plt.scatter(x, y)

# bayes model fitting:
def log_prior(theta):
    beta = theta
    return -1.5 * np.log(1 + beta ** 2)

def log_likelihood(beta, x, y, yerr):
    sigma = yerr
    y_model = beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior(theta, x, y, yerr):
    return log_prior(theta) + log_likelihood(theta, x, y, yerr)

def neg_log_prob_free(theta, x, y, yerr):
    return -log_posterior(theta, x, y, yerr)

# calculate probability of free model:
res = scipy.optimize.minimize(neg_log_prob_free, 0, args=(x, y, yerr), method='Powell')

plt.scatter(x, y)
plt.plot(x, x*res.x, '-', color='g')
print('The probability of this model is {0:.2g}'.format(np.exp(log_posterior(res.x, x, y, yerr))))
print('The optimized probability is {0:.4g}x'.format(np.float64(res.x)))

# bayes model fitting:
def log_likelihood_fixed(x, y, yerr):
    sigma = yerr
    y_model = -1/2*x

    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior_fixed(x, y, yerr):
    return log_likelihood_fixed(x, y, yerr)

plt.scatter(x, y)
plt.plot(x, -0.5*x, '-', color='purple')
print('The probability of this model is {0:.2g}'.format(np.exp(log_posterior_fixed(x, y, yerr))))

def model_selection(X, Y, Yerr, **kwargs):
    guess = kwargs.pop('guess', -0.5)

    # calculate probability of free model:
    res = scipy.optimize.minimize(neg_log_prob_free, guess, args=(X, Y, Yerr), method='Powell')
    
    # Compute error bars
    second_derivative = scipy.misc.derivative(log_posterior, res.x, dx=1.0, n=2, args=(X, Y, Yerr), order=3)
    cov_free = -1/second_derivative
    alpha_free = np.float64(res.x)
    log_free = log_posterior(alpha_free, X, Y, Yerr)
    
    # log goodness of fit for fixed models
    log_MAP = log_posterior_fixed(X, Y, Yerr)

    good_fit = log_free - log_MAP

    # occam factor - only the free model has a penalty
    log_occam_factor =(-np.log(2 * np.pi) + np.log(cov_free)) / 2 + log_prior(alpha_free)

    # give more standing to simpler models. but just a little bit!
    lg = log_free - log_MAP + log_occam_factor - 2
    return lg

model_selection(x, y, yerr)

n = 50  # number of data points
x = np.linspace(-10, 10, n)
yerr = np.abs(np.random.normal(0, 2, n))
y = x*-0.55 + np.random.normal(0, yerr, n)
plt.scatter(x, y)

model_selection(x, y, yerr)

def simulate_many_odds_ratios(n):
    """
    Given a number `n` of data points, simulate 1,000 data points drawn from a null model and an alternative model and
    compare the odds ratio for each.
    """
    iters = 1000
    lg1 = np.zeros(iters)
    lg2 = np.zeros(iters)

    for i in range(iters):
        x = np.linspace(-10, 10, n)
        yerr = np.abs(np.random.normal(0, 2, n))

        # simulate two models: only one matches the fixed model
        y1 = -0.5*x + np.random.normal(0, yerr, n)
        y2 = -0.46*x + np.random.normal(0, yerr, n)

        lg1[i] = model_selection(x, y1, yerr)
        
        m2 = model_selection(x, y2, yerr)
        # Truncate OR for ease of plotting
        if m2 < 10:
            lg2[i] = m2
        else:
            lg2[i] = 10
            
    return lg1, lg2

def make_figures(n):
    lg1, lg2 = simulate_many_odds_ratios(n)
    
    lg1 = np.sort(lg1)
    lg2 = np.sort(lg2)
    
    fifty_point1 = lg1[int(np.floor(len(lg1)/2))]
    fifty_point2 = lg2[int(np.floor(len(lg2)/2))]
    
    fig, ax = plt.subplots(ncols=2, figsize=(15, 7), sharey=True)
    fig.suptitle('Log Odds Ratio for n={0} data points'.format(n), fontsize=20)
    sns.kdeplot(lg1, label='slope=-0.5', ax=ax[0], cumulative=False)
    ax[0].axvline(x=fifty_point1, ls='--', color='k')
    ax[0].set_title('Data drawn from null model')
    ax[0].set_ylabel('Density')

    sns.kdeplot(lg2, label='slope=-0.46', ax=ax[1], cumulative=False)
    ax[1].axvline(x=fifty_point2, ls='--', color='k')
    ax[1].set_title('Data drawn from alternative model')
    fig.text(0.5, 0.04, 'Log Odds Ratio', ha='center', size=18)

    return fig, ax

fig, ax = make_figures(n=5)

fig, ax = make_figures(n=50)

