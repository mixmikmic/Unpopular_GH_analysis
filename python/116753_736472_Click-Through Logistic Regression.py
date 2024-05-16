# Import libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
get_ipython().magic('matplotlib inline')

# Read data
ad_data = pd.read_csv('advertising.csv')

# Check first few lines of data
ad_data.head()

ad_data.info()

ad_data.describe()

# Create histrogram with Pandas
ad_data['Age'].plot.hist(bins = 30)

# Create histogram using Seaborn
sns.set_style('whitegrid')
plt.rcParams["patch.force_edgecolor"] = True
sns.distplot(ad_data['Age'], bins = 30)

sns.jointplot(x = 'Age', y = 'Area Income', data = ad_data)

sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, kind = 'kde')

sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data, color = 'green')

sns.pairplot(ad_data, hue = 'Clicked on Ad', palette = 'bwr')

# Look at column names
ad_data.columns

# Separate data into x and y variables
x = ad_data.drop(['Clicked on Ad','Ad Topic Line','City','Country','Timestamp'], axis = 1) #Exclude target and string variables
y = ad_data['Clicked on Ad']

# Verify
x.columns

# Use x and y variables to split data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Import model
from sklearn.linear_model import LogisticRegression

# Create instance of the model
logmodel = LogisticRegression()

# Fit model on training set
logmodel.fit(x_train, y_train)

predictions = logmodel.predict(x_test)

# Import functions
from sklearn.metrics import classification_report, confusion_matrix

# Print classification report
print(classification_report(y_test, predictions))

# Print confusion matrix
print(confusion_matrix(y_test, predictions))

