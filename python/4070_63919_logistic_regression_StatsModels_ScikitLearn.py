import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14

df = pd.read_csv('train_blood.csv')
df.head()

df.corr(method='pearson')

df.drop(df.columns[[0, 3]], axis=1, inplace=True)
header = {u:v for u, v in zip(df.columns, ['since_last', 'donations', 'since_first', 'march2007'])}
df.rename(columns=header, inplace=True)
df.head()

df.describe()

n, bins, patches = plt.hist(df.donations, bins=15)
plt.xlabel('Number of Donations')
plt.ylabel('Count')

bp = plt.boxplot(df.donations)
plt.ylabel('Number of Donations')

plt.plot(df.donations, 'wo')
plt.xlabel('Volunteer')
plt.ylabel('Number of Donations')

plt.plot(df.since_first[df.march2007 == 0], df.donations[df.march2007 == 0], 'wo', label='0')
plt.plot(df.since_first[df.march2007 == 1], df.donations[df.march2007 == 1], 'ro', label='1')
plt.xlabel('Months Since First Donation')
plt.ylabel('Number of Donations')
plt.legend(loc='upper left')

from scipy.stats import pearsonr
print pearsonr(df.since_first[df.march2007 == 0], df.donations[df.march2007 == 0])
print pearsonr(df.since_first[df.march2007 == 1], df.donations[df.march2007 == 1])

import statsmodels.formula.api as smf
result = smf.logit(formula='march2007 ~ donations', data=df).fit()
print result.summary()

def logistic(b0, b1, x):
    linear = b0 + b1 * x
    return np.exp(linear) / (1.0 + np.exp(linear))

beta0, beta1 = result.params
donations = np.linspace(-40, 100, num=df.donations.count())

plt.plot(donations, logistic(beta0, beta1, donations), 'k-')
plt.plot(df.donations, logistic(beta0, beta1, df.donations), 'wo')
plt.xlabel('Number of Donations')
plt.ylabel('Probability for March Donation')

result.pred_table(threshold=0.5)

zeros, ones = df.march2007.count() - sum(df.march2007), sum(df.march2007)
print zeros, ones

# taken from https://github.com/statsmodels/statsmodels/issues/1577

def precision(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.

    Analagous to (absence of) Type I errors. Probability that a randomly
    selected document is classified correctly. I.e., no false negatives.
    """
    tp, fp, fn, tn = map(float, pred_table.flatten())
    return tp / (tp + fp)


def recall(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.

    Analagous to (absence of) Type II errors. Out of all the ones that are
    true, how many did you predict as true. I.e., no false positives.
    """
    tp, fp, fn, tn = map(float, pred_table.flatten())
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return np.nan


def accuracy(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.
    """
    tp, fp, fn, tn = map(float, pred_table.flatten())
    return (tp + tn) / (tp + tn + fp + fn)


def fscore_measure(pred_table, b=1):
    """
    For b, 1 = equal importance. 2 = recall is twice important. .5 recall is
    half as important, etc.
    """
    r = recall(pred_table)
    p = precision(pred_table)
    try:
        return (1 + b**2) * r*p/(b**2 * p + r)
    except ZeroDivisionError:
        return np.nan

print precision(result.pred_table())
print recall(result.pred_table())
print accuracy(result.pred_table())
print fscore_measure(result.pred_table())

import statsmodels.formula.api as smf
result = smf.logit(formula='march2007 ~ since_first + donations + since_last', data=df).fit()
print result.summary()

result.pred_table(threshold=0.5)

print precision(result.pred_table())
print recall(result.pred_table())
print accuracy(result.pred_table())
print fscore_measure(result.pred_table())

df['rate'] = pd.Series(df.donations / df.since_first, index=df.index)
df.head()

plt.plot(df.index, df.rate, 'wo')
plt.xlabel('Volunteer')
plt.ylabel('Rate of Donations')

bp = plt.boxplot(df.rate)

import statsmodels.formula.api as smf
result = smf.logit(formula='march2007 ~ since_last + rate', data=df).fit()
print result.summary()

result.pred_table(threshold=0.5)

print precision(result.pred_table())
print recall(result.pred_table())
print accuracy(result.pred_table())
print fscore_measure(result.pred_table())

def roc_curve(r):
    thresholds = np.linspace(0.2, 0.8, num=10)
    fpr = []
    tpr = []
    for threshold in thresholds:
        tp, fp, fn, tn = map(float, r.pred_table(threshold).flatten())
        if (fp + tn > 0 and tp + fn > 0):
          fpr.append(fp / (fp + tn))
          tpr.append(tp / (tp + fn))
    return fpr, tpr

fpr, tpr = roc_curve(result)
plt.plot(fpr, tpr, 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

from sklearn import linear_model
regr = linear_model.LogisticRegression()
columns = ['since_last', 'rate']
model = regr.fit(X=df[columns], y=df.march2007)
print type(model.coef_)
pd.DataFrame(zip(['intercept'] + columns, np.transpose(np.append(model.intercept_, model.coef_))))

df_test = pd.read_csv('test_blood.csv')
df_test.drop(df_test.columns[[0, 2]], axis=1, inplace=True)
header = {u:v for u, v in zip(df_test.columns, ['since_last', 'donations', 'since_first'])}
df_test.rename(columns=header, inplace=True)

# add new column
df_test['rate'] = pd.Series(df_test.donations / df_test.since_first, index=df_test.index)
df_test.head()

df_test.describe()

y_test = regr.predict(df_test[columns])

