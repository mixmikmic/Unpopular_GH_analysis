# Data obtained from here:
# https://raw.githubusercontent.com/johnmyleswhite/ML_for_Hackers/master/05-Regression/data/01_heights_weights_genders.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14

df = pd.read_csv('01_heights_weights_genders.csv')
df[4998:5002]

df.describe()

df.info()

df.dtypes

df.columns

plt.plot(df.Height[df.Gender == "Male"], df.Weight[df.Gender == "Male"], 'b+', label="Male")
plt.plot(df.Height[df.Gender == "Female"], df.Weight[df.Gender == "Female"], 'r+', label="Female")
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend(loc='upper left')

df.replace("Male", 0, inplace=True)
df.replace("Female", 1, inplace=True)
df[4998:5002]

import statsmodels.formula.api as smf
result = smf.logit(formula='Gender ~ Height + Weight', data=df).fit()
print result.summary()

import statsmodels.api as sm
kde_res = sm.nonparametric.KDEUnivariate(result.predict())
kde_res.fit()
plt.plot(kde_res.support, kde_res.density)
plt.fill_between(kde_res.support, kde_res.density, alpha=0.2)
plt.title("Distribution of predictions")

result.pred_table()

tp, fp, fn, tn = map(float, result.pred_table().flatten())
accuracy = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print accuracy
print precision
print recall

def roc_curve(r):
    thresholds = np.linspace(0.0, 1.0, num=100)
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
plt.title('Receiver Operating Characteristic')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

beta0, beta1, beta2 = result.params
x = np.linspace(50, 80)
y = -(beta0 + beta1 * x) / beta2
plt.plot(df.Height[df.Gender == 0], df.Weight[df.Gender == 0], 'b+', label="Male")
plt.plot(df.Height[df.Gender == 1], df.Weight[df.Gender == 1], 'r+', label="Female")
plt.plot(x, y, 'k-')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend(loc='upper left')

result = smf.glm('Gender ~ Height + Weight', data=df, family=sm.families.Binomial(sm.families.links.logit)).fit()
print result.summary()

