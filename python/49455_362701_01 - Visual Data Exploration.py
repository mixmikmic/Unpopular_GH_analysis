get_ipython().magic('matplotlib notebook')

import numpy as np
import matplotlib.pyplot as plt 

# Create the X data points as a numpy array 
X = np.linspace(-10, 10, 255)

# Compute two quadratic functions 
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 

plt.plot(X, Y1)
plt.plot(X, Y2)

# Create a new figure of size 8x6 points, using 72 dots per inch 
plt.figure(figsize=(8,6), dpi=72)

# Create a new subplot from a 1x1 grid 
plt.subplot(111)

# Create the data to plot 
X = np.linspace(-10, 10, 255)
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 

# Plot the first quadratic using a blue color with a continuous line of 1px
plt.plot(X, Y1, color='blue', linewidth=1.0, linestyle='-')

# Plot the second quadratic using a green color with a continuous line of 1px
plt.plot(X, Y2, color='green', linewidth=1.0, linestyle='-')

# Set the X limits 
plt.xlim(-10, 10)

# Set the X ticks 
plt.xticks(np.linspace(-10, 10, 9, endpoint=True))

# Set the Y limits 
plt.ylim(0, 350)

# Set the Y ticks 
plt.yticks(np.linspace(0, 350, 5, endpoint=True))

# Save the figure to disk 
plt.savefig("figures/quadratics.png")

# Create the data to plot 
# This data will be referenced for the next plots below
# For Jupyter notebooks, pay attention to variables! 

X = np.linspace(-10, 10, 255)
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 

from matplotlib.colors import ListedColormap

colors = 'bgrmyck'
fig, ax = plt.subplots(1, 1, figsize=(7, 1))
ax.imshow(np.arange(7).reshape(1,7), cmap=ListedColormap(list(colors)), interpolation="nearest", aspect="auto")
ax.set_xticks(np.arange(7) - .5)
ax.set_yticks([-0.5,0.5])
ax.set_xticklabels([])
ax.set_yticklabels([])

# plt.style.use('fivethirtyeight')

# Note that I'm going to use temporary styling so I don't mess up the notebook! 
with plt.style.context(('fivethirtyeight')):
    plt.plot(X, Y1)
    plt.plot(X, Y2)

# To see the available styles:
for style in plt.style.available: print("- {}".format(style))

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-")

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(Y1.min()*-1.1, Y2.max()*1.1)

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-", label="Y1")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-", label="Y2")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(Y1.min()*-1.1, Y2.max()*1.1)

plt.title("Two Quadratic Curves")
plt.legend(loc='best')

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-", label="Y1")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-", label="Y2")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(0, Y2.max()*1.1)

plt.title("Two Quadratic Curves")
plt.legend(loc='best')

# Annotate the blue line 
x = 6 
y = 2*x ** 2 + 10
plt.plot([x,x], [0, y], color='blue', linewidth=1.5, linestyle='--')
plt.scatter([x,], [y,], color='blue', s=50, marker='D')

plt.annotate(
    r'$2x^2+10={}$'.format(y), xy=(x,y), xycoords='data', xytext=(10,-50), 
    fontsize=16, textcoords='offset points',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
)

# Annotate the red line
x = -3
y = 3*x ** 2 + 50
plt.plot([x,x], [0, y], color='red', linewidth=1.5, linestyle='--')
plt.scatter([x,], [y,], color='red', s=50, marker='s')

plt.annotate(
    r'$3x^2+50={}$'.format(y), xy=(x,y), xycoords='data', xytext=(10,50), 
    fontsize=16, textcoords='offset points',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
)

import pandas as pd

# Create a random timeseries object 
ts = pd.Series(np.random.randn(365), index=pd.date_range('1/1/2010', periods=365))
ts = ts.cumsum()
ts.plot() 

df = pd.DataFrame(np.random.randn(365, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
df.plot();

df = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
df['A'] = pd.Series(list(range(len(df))))
df.plot(x='A', y='B')

df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df2.plot(kind='bar')

df2.plot.bar(stacked=True)

df2.plot(kind='area')

df2.plot.area(stacked=False)

df3 = pd.DataFrame(np.random.rand(25, 4), columns=['a', 'b', 'c', 'd'])
ax = df3.plot.scatter(x='a', y='b', color='r', label="B")
df3.plot.scatter(x='a', y='c', color='c', ax=ax, label="C")
df3.plot.scatter(x='a', y='d', color='g', ax=ax, label="D")

# Add new dimensions such as color and size based on other attributes in the data frame
# This creates a bubble plot, points sized based on the 'C' attribute and colors based on 'D'.
df3.plot.scatter(x='a', y='b', c=df3['d'], s=df3['c']*200);

df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df['b'] = df['b'] + np.arange(1000)
df.plot.hexbin(x='a', y='b', gridsize=25)

# Load the data
import os 

DATA = os.path.join("data", "wheat", "seeds_dataset.txt")

FEATURES  = [
    "area",
    "perimeter",
    "compactness",
    "length",
    "width",
    "asymmetry",
    "groove",
    "label"
]

LABEL_MAP = {
    1: "Kama",
    2: "Rosa",
    3: "Canadian",
}

# Read the data into a DataFrame
df = pd.read_csv(DATA, sep='\s+', header=None, names=FEATURES)

# Convert class labels into text
for k,v in LABEL_MAP.items():
    df.ix[df.label == k, 'label'] = v

# Describe the dataset
print(df.describe())

# Determine the shape of the data
print("{} instances with {} features\n".format(*df.shape))

# Determine the frequency of each class
print(df.groupby('label')['label'].count())

from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(9,9), diagonal='kde')

from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(9,9))
parallel_coordinates(df, 'label')

from pandas.tools.plotting import radviz
plt.figure(figsize=(9,9))
radviz(df, 'label')

get_ipython().magic('matplotlib inline')
import os
import pandas as pd
import seaborn as sns

IRIS = os.path.join("data", "iris.csv")
data = pd.read_csv(IRIS)

sns.set_style('darkgrid')
sns.set_palette('deep')
sns.set_context('notebook')

sns.pairplot(data, hue='class', diag_kind="kde", size=3)

sns.distplot(data['sepal width'], rug=True)

sns.jointplot("petal length", "petal width", data=data, kind='reg', size=6)

sns.boxplot(x='petal length', data=data)

sns.boxplot(data=data)

sns.violinplot(data=data)

sns.swarmplot(data=data)

ax = sns.boxplot(data=data)
ax = sns.swarmplot(data=data)

sns.lmplot(x="sepal width", y="sepal length", hue="class", data=data)

sns.lmplot(x="sepal width", y="sepal length", col="class", data=data)

sns.barplot(data=data)

import numpy as np
sns.barplot(data=data, estimator=np.median)

