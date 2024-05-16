from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest

import numpy as np
str_lines = sc.textFile('/Users/jhalverson/data_science/machine_learning/wine.csv')
data_labels = str_lines.map(lambda line: int(line.split(',')[0]) - 1)
data_features = str_lines.map(lambda line: np.array([float(x) for x in line.split(',')[1:]]))
print 'Total records:', data_features.count()

data = data_labels.zip(data_features)
data = data.map(lambda x: LabeledPoint(x[0], [x[1]]))
data.take(2)

train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)
train_data.persist(StorageLevel.DISK_ONLY)
train_data.map(lambda x: x.label).countByValue().items()

model = RandomForest.trainClassifier(train_data, numClasses=3, categoricalFeaturesInfo={}, numTrees=100,
                                     featureSubsetStrategy='sqrt', impurity='gini', maxBins=32)

test_data_features = test_data.map(lambda x: x.features)
test_data_labels = test_data.map(lambda x: x.label)
predictions = model.predict(test_data_features)

ct = 0
for true, pred in zip(test_data_labels.collect(), predictions.collect()):
    if true == pred: ct += 1
print float(ct) / test_data_labels.count()

