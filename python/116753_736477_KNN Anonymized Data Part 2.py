import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

get_ipython().magic('matplotlib inline')

df = pd.read_csv('KNN_Project_Data')

df.head()

sns.pairplot(df, hue = 'TARGET CLASS', palette = 'coolwarm')

df.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS', axis = 1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_feat.head(3)

# Import
from sklearn.model_selection import train_test_split

x = df_feat
y = df['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(x_train,y_train)

# Predict values using knn model
pred = knn.predict(x_test)

# Import
from sklearn.metrics import confusion_matrix, classification_report

# Print confusion matrix
print(confusion_matrix(y_test, pred))

# Print classification report
print(classification_report(y_test,pred))

error_rate = []

for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i!=y_test))

plt.figure(figsize = (10,6))
plt.plot(range(1,60),error_rate, color = 'blue',linestyle = '--', marker = 'o',
        markerfacecolor = 'orange', markersize = 10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Retrain model
knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(x_train, y_train)

# Make new predictions
pred = knn.predict(x_test)

# Print confusion matrix
print(confusion_matrix(y_test, pred))

# New line
print('\n')

# Print classification report
print(classification_report(y_test,pred))

