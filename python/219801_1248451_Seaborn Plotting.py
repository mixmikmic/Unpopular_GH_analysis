import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

video_games_raw = pd.read_csv('vgsales.csv', parse_dates=['Year'])
video_games_raw.describe()

video_games_raw['NA_Sales'].quantile(0.95)

video_games_raw['EU_Sales'].quantile(0.95)

video_games = video_games_raw[video_games_raw['NA_Sales'] < 1.06]

video_games = video_games_raw[video_games_raw['EU_Sales'] < 0.63]

video_games.describe()

#PLOT 1
g = sns.lmplot(y='NA_Sales', # Variable 1.
               x='EU_Sales', # Variable 2.
               data=video_games, # Data
               fit_reg=False, # If set to true, plots a regression line.
               scatter_kws={'alpha':0.4}) # Set points to semi-transparent to see overlaping points.
g.set_ylabels("North America Sales")
g.set_xlabels("European Union Sales")
plt.title('North America Sales vs. European Union Sales')
plt.show()

#PLOT 2
g = sns.lmplot(y='NA_Sales', # Variable 1.
               x='EU_Sales', # Variable 2.
               data=video_games, # Data
               fit_reg=True, # If set to true, plots a regression line.
               scatter_kws={'alpha':0.4}) # Set points to semi-transparent to see overlaping points.
g.set_ylabels("North America Sales")
g.set_xlabels("European Union Sales")
plt.title('North America Sales vs. European Union Sales')
plt.show()

#PLOT 3
g = sns.jointplot(x="EU_Sales", y="NA_Sales", data=video_games, kind="kde")
plt.show()

#PLOT 1
plt.figure(figsize=[12,12])
plt.hist(video_games['NA_Sales'], bins=50)
# plt.yticks(np.arange(0, 17000, 250), rotation='horizontal')
plt.title('Distribution of National Sales')
plt.xlabel('Sales')
plt.ylabel('Number of Occurrences')
plt.show()

#PLOT 2
plt.figure(figsize=[12,12])
plt.boxplot(video_games['NA_Sales'])
plt.title('Boxplot of National Sales')
plt.show()

#PLOT 3
plt.figure(figsize=[12,12])
sns.distplot(video_games['NA_Sales'])
plt.title('Density Plot of National Sales')
plt.show()

#PLOT 4
sns.kdeplot(video_games['NA_Sales'], shade=True, cut=0)
sns.rugplot(video_games['NA_Sales'])
plt.title('Density and Rug Plot of North America Sales')
plt.show()

#PLOT 1
# Comparing groups using boxplots.
plt.figure(figsize=[12,5])
ax = sns.boxplot(x='Genre',y='EU_Sales', data=video_games)  
plt.title('Boxplots for EU Sales by Genre')
sns.despine(offset=10, trim=True)
ax.set(xlabel='Genre', ylabel='EU Sales')
plt.show()

#PLOT 2
# Setting the overall aesthetic.
sns.set(style="darkgrid")
g = sns.factorplot(x="Genre", y="EU_Sales", data=video_games,
                   size=6, kind="bar", palette="pastel", ci=95)
g.despine(left=True)
g.set_ylabels("EU Sales")
g.set_xlabels("Genre")
plt.title('EU Sales by Genre')
plt.xticks(rotation='vertical')
plt.show()

#PLOT 3
# Setting the overall aesthetic.
sns.set(style="whitegrid")

g = sns.factorplot(x="Genre", y="EU_Sales", data=video_games,
                   size=6, kind="point", palette="pastel",ci=95,dodge=True,join=False)
g.despine(left=True)
g.set_ylabels("EU Sales")
g.set_xlabels("Genre")
plt.xticks(rotation='vertical')
plt.title('EU Sales by Genre')
plt.show()

#PLOT 4
sns.stripplot(x="Genre", y="EU_Sales", data=video_games)
plt.xticks(rotation='vertical')
plt.title('Strip Plot of EU Sales by Genre')
plt.show()

#PLOT 5
sns.stripplot(x="Genre", y="EU_Sales", data=video_games, jitter=True)
plt.xticks(rotation='vertical')
plt.title('Strip Plot (with Jitter) of EU Sales by Genre')
plt.show()

#PLOT 6
sns.swarmplot(x="Genre", y="EU_Sales", data=video_games)
plt.xticks(rotation='vertical')
plt.title('Swarm Plot of EU Sales by Genre')
plt.show()



