from sklearn import datasets
iris = datasets.load_iris()

#print(iris)
# X = inputs for the classifier
X = iris.data

# y = ouput 
y = iris.target
print(y.size)

# We can either manually partition dataset into test and training dataset or either use cross validation
from sklearn.cross_validation import train_test_split

#help(train_test_split)
# Using half of the dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)


# Checking accuracy of the classifier
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# Checking accuracy of the classifier
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

