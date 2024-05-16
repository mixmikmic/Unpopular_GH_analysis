import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')

# Set the plot background.
sns.set_style('darkgrid')

df = pd.read_csv('IBM HR Analytics Employee Attrition.csv')

# Determine which categorical and continuous variables to use.
df.head()

# Determine how big my dataset is.
df.shape

print(list(df))

# Checking out the initial correlation of the data set presented.
corrmat = df.corr()

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Create a new dataframe to have only the data I want to analyze. 
df2 = pd.DataFrame()
df2 = df.loc[:, ['Age', 'Attrition', 'Department', 'Gender', 'YearsAtCompany', 'MonthlyIncome']]

df2.head()

# Next let see what initial correlation we can glean from the existing continuous data. 
g = sns.PairGrid(df2.dropna(), diag_sharey=False)
# Scatterplot.
g.map_upper(plt.scatter, alpha=.5)
# Fit line summarizing the linear relationship of the two variables.
g.map_lower(sns.regplot, scatter_kws=dict(alpha=.5))
# Give information about the univariate distributions of the variables.
g.map_diag(sns.kdeplot, lw=4)
plt.show()

# Lets look at the heatmap one more time
corrmat = df2.corr()

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

print(corrmat)

# Plot categorical data with continuous data using box - plots.
#Make a four-panel plot.
sns.boxplot(x = df2['Attrition'], y = df2['YearsAtCompany'])
plt.show()

sns.boxplot(x = df2['Attrition'], y = df2['MonthlyIncome'])
plt.show()

sns.boxplot(x = df2['Attrition'], y = df2['Age'])
plt.show()

# It's time to check if the distribution of my continuous data has a normal distribution or not. 
s = df.groupby('Age').count()['EmployeeCount']
g = sns.barplot(s.index, s, palette='GnBu_d')
g.figure.set_size_inches(10,10)
g.set_title("Age Distribution")
sns.set_style('darkgrid')
plt.show()

# Years at Company
s = df.groupby('YearsAtCompany').count()['EmployeeCount']
g = sns.barplot(s.index, s, palette='GnBu_d')
g.figure.set_size_inches(10,10)
g.set_title("Years at Company Distribution")
plt.show()

df2.groupby('Attrition').describe()

# Two Categorical Variables (Gender & Attrition)

# Plot counts for each combination of levels.
sns.countplot(y="Gender", hue="Attrition", data=df, palette="Greens_d")
plt.show()

# Table of counts
counttable = pd.crosstab(df['Gender'], df['Attrition'])
print(counttable)

# Test will return a chi-square test statistic and a p-value. Like the t-test,
# the chi-square is compared against a distribution (the chi-square
# distribution) to determine whether the group size differences are large
# enough to reflect differences in the population.
print(stats.chisquare(counttable, axis=None))

# Try different graphs to standardize MonthlyIncome first 
# since there were no graphs for MonthlyIncome. 

# Making a four-panel plot.
fig = plt.figure()

fig.add_subplot(221)
plt.hist(df['MonthlyIncome'].dropna())
plt.title('Raw')

# Feat 1 Log of Monthly Income to see if the values would become a normal distribution.
df['feat1'] = np.log(df['MonthlyIncome'])
fig.add_subplot(222)
plt.hist(np.log(df['MonthlyIncome'].dropna()))
plt.title('Log')

# Feat2, Sqrt of Monthly Income to see if values can become normalized.
df['feat2'] = np.sqrt(df['MonthlyIncome'])
fig.add_subplot(223)
plt.hist(np.sqrt(df['MonthlyIncome'].dropna()))
plt.title('Square root')

# Feat3, Inverse of Monthly Income to see if values can become normalized.
df['feat3'] = (1/df['MonthlyIncome'])
ax3=fig.add_subplot(224)
plt.hist(1/df['MonthlyIncome'].dropna())
plt.title('Inverse')
plt.tight_layout()
plt.show()

# Feat4, Square of Monthly Income to see if values can become normalized.
df['feat4'] = np.square(df['MonthlyIncome'])

# Feature 5. Standardize my three original features since the earlier Monthly Income
# didn't seem to do much.
from sklearn.preprocessing import StandardScaler
features = ['YearsAtCompany', 'Age', 'MonthlyIncome']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['Attrition']].values
# Standardizing the features
X = StandardScaler().fit_transform(x)

# The NumPy covariance function assumes that variables are represented by rows,
# not columns, so we transpose X.
Xt = X.T
Cx = np.cov(Xt)
print('Covariance Matrix:\n', Cx)

#Print the eigenvectirs and eigenvalues.
eig_vals, eig_vecs = np.linalg.eig(Cx)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Use sklearn to perform PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Attrition']]], axis = 1)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Yes', 'No']
colors = ['r', 'g']
for target, color in zip(targets,colors): 
    indicesToKeep = finalDf['Attrition'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 40)
ax.legend(targets)
ax.grid()

# Feature 6 Monthly Income divide by age so that we can see the avg income of each age.
df2['feat6'] = df['MonthlyIncome']/ df['Age'] 

#Feature 7: Group ages together, as more datapoints in a group should prevent defaults from 
# skewing rates in small populations

# Set a default value
df2['feat7'] = '0'
# Set Age_Group value for all row indexes which Age is LT 18
df2['feat7'][df2['Age'] <= 18] = 'LTE 18'
# Same procedure for other age groups
df2['feat7'][(df2['Age'] > 18) & (df2['Age'] <= 30)] = '19-30'
df2['feat7'][(df2['Age'] > 30) & (df2['Age'] <= 45)] = '31-45'
df2['feat7'][(df2['Age'] > 46) & (df2['Age'] <= 64)] = '46-64' 
df2['feat7'][(df2['Age'] > 65)] = '65+'



