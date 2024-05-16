from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

import csv
str_lines = sc.textFile("/Users/jhalverson/data_science/project_blood_donations/train_blood.csv")
train_rdd = str_lines.mapPartitions(lambda x: csv.reader(x)).filter(lambda x: 'Months' not in x[1]).map(lambda x: map(int, x))
str_lines = sc.textFile("/Users/jhalverson/data_science/project_blood_donations/test_blood.csv")
test_rdd = str_lines.mapPartitions(lambda x: csv.reader(x)).filter(lambda x: 'Months' not in x[1]).map(lambda x: map(int, x))

from pyspark.mllib.stat import Statistics
Statistics.corr(train_rdd, method='pearson')

train_features = train_rdd.map(lambda x: (x[1], x[2], x[4], float(x[4]) / x[2]))
train_labels = train_rdd.map(lambda x: x[-1])
test_features = test_rdd.map(lambda x: (x[1], x[2], x[4], float(x[4]) / x[2]))
test_labels = test_rdd.map(lambda x: x[-1])

train_features.take(2)

train_labels.take(2)

train_labels.countByValue().items()

# the code fails when the scaler is fit to the training data then applied to the test data
stdsc1 = StandardScaler(withMean=True, withStd=True).fit(train_features)
train_features_std = stdsc1.transform(train_features)
stdsc2 = StandardScaler(withMean=True, withStd=True).fit(test_features)
test_features_std = stdsc2.transform(test_features)

from pyspark.mllib.stat import Statistics
train_features_std_stats = Statistics.colStats(train_features_std)
print 'train means:', train_features_std_stats.mean()
print 'train variances:', train_features_std_stats.variance()
test_features_std_stats = Statistics.colStats(test_features_std)
print 'test means:', test_features_std_stats.mean()
print 'test means:', test_features_std_stats.variance()

import numpy as np
trainData = train_labels.zip(train_features_std)
trainData = trainData.map(lambda x: LabeledPoint(x[0], np.asarray(x[1:]))).cache()
trainData.take(5)

model = LogisticRegressionWithLBFGS.train(trainData, regParam=0.75)
model.clearThreshold()    

print model.weights, model.intercept

testData = test_rdd.map(lambda x: x[0]).zip(test_features_std.map(lambda x: model.predict(x)))

f = open('halverson_logistic_regression_may13_2016.dat', 'w')
f.write(',Made Donation in March 2007\n')
for volunteer_id, prob in testData.collect():
  f.write('%d,%.3f\n' % (volunteer_id, prob))
f.close()

