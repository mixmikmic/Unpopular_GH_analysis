# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

# Import library
import seaborn as sns

# Load dataset
iris = sns.load_dataset('iris')
iris.head()

import pandas as pd
import numpy as np
import sklearn as sns
get_ipython().magic('matplotlib inline')

sns.pairplot(iris, hue = 'species', palette= 'Dark2')

setosa = iris[iris['species']== 'setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'], cmap = 'coolwarm', shade = True, shade_lowest = False)

# Import function
from sklearn.model_selection import train_test_split

# Set variables
x = iris.drop('species', axis =1)
y = iris['species']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Import model
from sklearn.svm import SVC

# Create instance of model
svc_model = SVC()

# Fit model to training data
svc_model.fit(x_train, y_train)

# Predictions
predictions = svc_model.predict(x_test)

# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
print(confusion_matrix(y_test, predictions))

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions))

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

# Create GridSearchCV object
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
                    
# Fit to training data
grid.fit(x_train,y_train)

# Confusion matrix
print(confusion_matrix(y_test,predictions))

# New line
print('\n')

# Classification Report
print(classification_report(y_test,predictions))

