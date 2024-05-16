# install relevant modules
import numpy as np

# scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# import load_iris function from datasets module
from sklearn.datasets import load_iris

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt
# allow plots to appear within the notebook
get_ipython().magic('matplotlib inline')
import seaborn as sns

# save "bunch" object containing iris dataset and its attributes into iris_df
iris_df = load_iris()
type(iris_df)

# Look into the features 
print (iris_df.feature_names)
print (iris_df.data[0:3, :])
print(type(iris_df.data))

# Look into the labels
print (iris_df.target_names)
print (iris_df.target[:3])
print(type(iris_df.target))

# store feature matrix in X and label vector in y
X = iris_df.data
y = iris_df.target
# print and check shapes of X and y
print("shape of X: ", X.shape, "& shape of y: ", y.shape)

# KNN classification 
# Instantiate the estimator 
knn1 = KNeighborsClassifier(n_neighbors=1)
knn5 = KNeighborsClassifier(n_neighbors=5)

# Train the model
# output displays the default values
knn1.fit(X, y)
knn5.fit(X, y)

# Predict the response
X_new = [[3, 4, 5, 2], [5, 2, 3, 2]]
print("n_neighbors=1 predicts: ", knn1.predict(X_new))
print("n_neighbors=5 predicts: ", knn5.predict(X_new))

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
# output displays the default values
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)

# store the predicted response values
y_pred_knn1 = knn1.predict(X)
y_pred_knn5 = knn5.predict(X)
y_pred_logreg = logreg.predict(X)

# compute classification accuracy for the logistic regression model
print("Accuracy of KNN with n_neighbors=1: ", metrics.accuracy_score(y, y_pred_knn1))
print("Accuracy of KNN with n_neighbors=5: ", metrics.accuracy_score(y, y_pred_knn5))
print("Accuracy of logistic regression: ", metrics.accuracy_score(y, y_pred_logreg))

# Splitting the data in 75% training data and 25% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
print("shape of X_train: ", X_train.shape, "& shape of y_train: ", y_train.shape)
print("shape of X_test: ", X_test.shape, "& shape of y_test: ", y_test.shape)

# Instantiate the estimators 
knn1 = KNeighborsClassifier(n_neighbors=1)
knn5 = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression()

# Train the models
# output displays the default values
logreg.fit(X_train, y_train)
knn1.fit(X_train, y_train)
knn5.fit(X_test, y_test)
print('\n')

# Predictions
y_pred_knn1 = knn1.predict(X_test)
y_pred_knn5 = knn5.predict(X_test)
y_pred_logreg = logreg.predict(X_test)

# compute classification accuracy for the logistic regression model
print("Accuracy of KNN with n_neighbors=1: ", metrics.accuracy_score(y_test, y_pred_knn1))
print("Accuracy of KNN with n_neighbors=5: ", metrics.accuracy_score(y_test, y_pred_knn5))
print("Accuracy of logistic regression: ", metrics.accuracy_score(y_test, y_pred_logreg))

# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26, 2))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

# plot the relationship between K and testing accuracy
plt.rcParams.update({'font.size': 18})
plt.plot(k_range, scores, 'ro', linewidth=2.0, linestyle="-")
plt.xlabel('K')
plt.ylabel('Accuracy')

