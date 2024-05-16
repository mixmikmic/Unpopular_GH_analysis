from sklearn import metrics
import numpy as np
import sklearn.datasets
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split

# clear string
def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

# because of sklean.datasets read a document as a single element
# so we want to split based on new line
def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        # python3, if python2, just remove list()
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget

# you can change any encoding type
trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset)
print (trainset.target_names)
print (len(trainset.data))
print (len(trainset.target))

# bag-of-word
bow = CountVectorizer().fit_transform(trainset.data)

#tf-idf, must get from BOW first
tfidf = TfidfTransformer().fit_transform(bow)

#hashing, default n_features, probability cannot divide by negative
hashing = HashingVectorizer(non_negative = True).fit_transform(trainset.data)

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

mod_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = mod_huber.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))

train_X, test_X, train_Y, test_Y = train_test_split(tfidf, trainset.target, test_size = 0.2)

mod_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = mod_huber.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))

train_X, test_X, train_Y, test_Y = train_test_split(hashing, trainset.target, test_size = 0.2)

mod_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = mod_huber.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

svm = SGDClassifier(penalty = 'l2', alpha = 1e-3, n_iter = 10).fit(train_X, train_Y)
predicted = svm.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

sq_hinge = SGDClassifier(loss = 'squared_hinge', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = sq_hinge.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

perceptron = SGDClassifier(loss = 'perceptron', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = perceptron.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

mod_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = mod_huber.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))

# get probability for first 2 sentence in our dataset
print(trainset.data[:2])
print(trainset.target[:2])
print(mod_huber.predict_proba(bow[:2, :]))

