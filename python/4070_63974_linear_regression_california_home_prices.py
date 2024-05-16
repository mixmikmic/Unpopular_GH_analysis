from pyspark.sql import Row

str_lines = sc.textFile('Sacramentorealestatetransactions.csv')
homes = str_lines.map(lambda x: x.split(','))
homes.take(2)

header = homes.first()
homes = homes.filter(lambda line: line != header)
homes.count()

# this function fails for the homes data but works in the example below
def makeRow(x):
    s = ''
    for i, item in enumerate(header):
        s += item + '=x[' + str(i) + '],'
    return eval('Row(' + s[:-1] + ')')

r = makeRow(range(20))
print r.baths

def makeRow2(x):
    return Row(street=x[0], city=x[1], zipcode=int(x[2]), beds=int(x[4]), baths=int(x[5]), sqft=int(x[6]), price=int(x[9]))

df = homes.map(makeRow2).toDF()
df.printSchema()

df.show()

df.select('city', 'beds').show(5)

df.groupBy('beds').count().show()

df.describe().show()

df = df[df.baths > 0]
df = df[df.beds > 0]
df = df[df.sqft > 0]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

pf = df.toPandas()

plt.plot(pf['sqft'], pf['price'], 'wo')
plt.xlabel('Square feet')
plt.ylabel('Price')

from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint

df = df.select('price','baths','beds','sqft')
data_features = df.map(lambda x: x[1:])
data_features.take(5)

from pyspark.mllib.feature import StandardScaler
stdsc = StandardScaler(withMean=False, withStd=True).fit(data_features)
data_features_std = stdsc.transform(data_features)

from pyspark.mllib.stat import Statistics
data_features_std_stats = Statistics.colStats(data_features_std)
print 'train means:', data_features_std_stats.mean()
print 'train variances:', data_features_std_stats.variance()

transformed_data = df.map(lambda x: x[0]).zip(data_features_std)
transformed_data = transformed_data.map(lambda x: LabeledPoint(x[0], [x[1]]))
transformed_data.take(5)

train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=1234)

linearModel = LinearRegressionWithSGD.train(train_data, iterations=1000, step=0.25, intercept=False)
print linearModel.weights

from pyspark.mllib.evaluation import RegressionMetrics
prediObserRDDin = train_data.map(lambda row: (float(linearModel.predict(row.features[0])), row.label))
metrics = RegressionMetrics(prediObserRDDin)
print metrics.r2

prediObserRDDout = test_data.map(lambda row: (float(linearModel.predict(row.features[0])), row.label))
metrics = RegressionMetrics(prediObserRDDout)
print metrics.rootMeanSquaredError

