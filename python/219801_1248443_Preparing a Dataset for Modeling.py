import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')

#Read the data in
survey_raw = pd.read_csv('student survey responses.csv')

#Get a feel for number of observations, columns
survey_raw.shape

survey_raw.head(5)

#Since there are a lot of columns, change default. In the describe() method, pass the 'include' argument so that we can understand 
#any categorical variables as well

pd.options.display.max_columns = 150
survey_raw.describe(include='all')

#Two ways to get all missing values in the DataFrame

#1 Chain isnull() and sum() 
missing_values_count = survey_raw.isnull().sum()

# 2 Define a function
def missing(x):
  return sum(x.isnull())

print("Missing values per column:")
print(survey_raw.apply(missing, axis=0))

#Calculate the variance across the DataFrame

survey_raw.var().nlargest(25)

#Create a clean DataFrame with dropped NaN's

survey_clean = survey_raw.dropna(subset=['Gender', 'Smoking', 'Alcohol', 'Healthy eating',
                                         'Hypochondria', 'Health', 'Spending on healthy eating'])

#Subset the 'clean' DataFrame

survey_filtered = survey_clean.loc[((survey_clean['Gender'] == 'male') | (survey_clean['Gender'] == 'female')),
    ['Gender', 'Smoking', 'Alcohol', 'Healthy eating', 'Hypochondria', 'Health', 'Spending on healthy eating']]

#Rename columns so that we can interpret better

survey_filtered.rename(columns={'Healthy eating': 'Healthy Lifestyle', 'Health': 'Worry About Health'}, inplace=True)

#Scatterplot Matrix

#First, because our continuous variables overlap (1-5 scale), make a copy of the data to add jitter 
#to and plot.

survey_filtered_jittered = survey_filtered.loc[:, 'Healthy Lifestyle':'Spending on healthy eating']

# Making the random noise.
jitter = pd.DataFrame(
    np.random.uniform(-.3, .3, size=(survey_filtered_jittered.shape)),
    columns=survey_filtered_jittered.columns
)
# Combine the data and the noise.
survey_filtered_jittered = survey_filtered_jittered.add(jitter)

# Declare that you want to make a scatterplot matrix.
g = sns.PairGrid(survey_filtered_jittered.dropna(), diag_sharey=False)
# Scatterplot.
g.map_upper(plt.scatter, alpha=.5)
# Fit line summarizing the linear relationship of the two variables.
g.map_lower(sns.regplot, scatter_kws=dict(alpha=0))
# Give information about the univariate distributions of the variables.
g.map_diag(sns.kdeplot, lw=3, shade=True)
plt.show()

# Make the correlation matrix.
corr_matrix = survey_filtered.corr()
print(corr_matrix)

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn.
sns.heatmap(corr_matrix, vmax=.8, square=True, cmap='YlGnBu')
plt.show()

# Restructure the data so we can use FacetGrid rather than making a boxplot
# for each variable separately.
survey_filtered_continuous = survey_filtered.loc[:, ['Gender', 'Healthy Lifestyle', 'Hypochondria', 'Worry About Health', 'Spending on healthy eating']]
survey_filtered_melted = survey_filtered_continuous
survey_filtered_melted = pd.melt(survey_filtered_melted, id_vars=['Gender'])

g = sns.FacetGrid(survey_filtered_melted, col="variable", size=6, aspect=.5)
g = g.map(sns.boxplot, "Gender", "value")
plt.show()

# Descriptive statistics by group.
print(survey_filtered_continuous.groupby('Gender').describe())

# Test whether group differences are significant.
for col in survey_filtered_continuous.loc[:,'Healthy Lifestyle':'Spending on healthy eating'].columns:
    print(col)
    print(stats.ttest_ind(
        survey_filtered_continuous[survey_filtered_continuous['Gender'] == 'male'][col],
        survey_filtered_continuous[survey_filtered_continuous['Gender'] == 'female'][col]))

#Look at Gender vs. Smoking

# Plot counts for each combination of levels.
plt.figure(figsize=[12,12])
plt.yticks(np.arange(0, 250, 10))
sns.countplot(x='Gender', hue="Smoking", data=survey_filtered, palette="Reds_d")
plt.show()

# Table of counts
counttable_smoking = pd.crosstab(survey_filtered['Gender'], survey_filtered['Smoking'])
print(counttable_smoking)

# Test will return a chi-square test statistic and a p-value. Like the t-test,
# the chi-square is compared against a distribution (the chi-square
# distribution) to determine whether the group size differences are large
# enough to reflect differences in the population.
print(stats.chisquare(counttable_smoking, axis=None))

#Look at Gender vs. Alcohol

# Plot counts for each combination of levels.
plt.figure(figsize=[12,12])
plt.yticks(np.arange(0, 425, 25))
sns.countplot(x='Gender', hue="Alcohol", data=survey_filtered, palette="Blues_d")
plt.show()

# Table of counts
counttable_alcohol = pd.crosstab(survey_filtered['Gender'], survey_filtered['Alcohol'])
print(counttable_alcohol)

# Test will return a chi-square test statistic and a p-value. Like the t-test,
# the chi-square is compared against a distribution (the chi-square
# distribution) to determine whether the group size differences are large
# enough to reflect differences in the population.
print(stats.chisquare(counttable_alcohol, axis=None))

#Make dummies and create DataFrame to start collecting features

features = pd.get_dummies(survey_filtered['Gender'])

#Start creating features above. First is 'Drinker'

features['Drinker'] = np.where((survey_filtered['Alcohol'].isin(['social drinker', 'drinks a lot'])), 1, 0)
print(pd.crosstab(features['Drinker'], survey_filtered['Gender']))

#Create 'Smoker' feature

features['Smoker'] = np.where((survey_filtered['Smoking'].isin(['former smoker', 'current smoker'])), 1, 0)
print(pd.crosstab(features['Smoker'], survey_filtered['Gender']))

#Create 'Drinker Healthy Lifestyle' feature

features['Drinker Healthy Lifestyle'] = np.where((survey_filtered['Alcohol'].isin(['social drinker', 'drinks a lot'])) & (survey_filtered['Healthy Lifestyle'] >= 3), 1, 0)
print(pd.crosstab(features['Drinker Healthy Lifestyle'], survey_filtered['Gender']))

#Create 'Smoker Healthy Lifestyle' feature

features['Smoker Healthy Lifestyle'] = np.where((survey_filtered['Smoking'] == 'current smoker') & (survey_filtered['Healthy Lifestyle'] >= 3), 1, 0)
print(pd.crosstab(features['Smoker Healthy Lifestyle'], survey_filtered['Gender']))

#Create 'Non Drinker Unhealthy Lifestyle' feature

features['Non Drinker Unhealthy Lifestyle'] = np.where((survey_filtered['Alcohol'] == 'never') & (survey_filtered['Healthy Lifestyle'] <= 3), 1, 0)
print(pd.crosstab(features['Non Drinker Unhealthy Lifestyle'], survey_filtered['Gender']))

#Create 'Non Smoker Unhealthy Lifestyle' feature

features['Non Smoker Unhealthy Lifestyle'] = np.where((survey_filtered['Smoking'] == 'never smoked') & (survey_filtered['Healthy Lifestyle'] <= 3), 1, 0)
print(pd.crosstab(features['Non Smoker Unhealthy Lifestyle'], survey_filtered['Gender']))

#Create 'Drinker Not Worried About Health' feature

features['Drinker Not Worried About Health'] = np.where((survey_filtered['Alcohol'].isin(['social drinker', 'drinks a lot'])) & (survey_filtered['Worry About Health'] <= 3), 1, 0)
print(pd.crosstab(features['Drinker Not Worried About Health'], survey_filtered['Gender']))

#Create 'Smoker Not Worried About Health' feature

features['Smoker Not Worried About Health'] = np.where((survey_filtered['Smoking'] == 'current smoker') & (survey_filtered['Worry About Health'] <= 3), 1, 0)
print(pd.crosstab(features['Smoker Not Worried About Health'], survey_filtered['Gender']))

#Create 'Healthy Lifestyle Doesn't Spend on Healthy Eating' feature

features['Healthy Lifestyle Doesnt Spend on Healthy Eating'] = np.where((survey_filtered['Spending on healthy eating'] <= 3) & (survey_filtered['Healthy Lifestyle'] >= 3), 1, 0)
print(pd.crosstab(features['Healthy Lifestyle Doesnt Spend on Healthy Eating'], survey_filtered['Gender']))

#Create 'Healthy Lifestyle and Hypochondriac' Feature

features['Healthy Lifestyle and Hypochondriac'] = np.where((survey_filtered['Hypochondria'] >= 3) & (survey_filtered['Healthy Lifestyle'] >= 3), 1, 0)
print(pd.crosstab(features['Healthy Lifestyle and Hypochondriac'], survey_filtered['Gender']))

