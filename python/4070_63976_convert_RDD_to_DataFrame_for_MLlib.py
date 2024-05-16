df = spark.createDataFrame([(["a", "b", "c"],)], ["words"])
df.show()

import pandas as pd
x = df.toPandas()
x

df = spark.createDataFrame([(["a", "b", "c"],), (["e", "f", "g"],)], ["words"]).show()

df = spark.createDataFrame([(["a", "b", "c"],), (["e", "f", "g"],)], schema=["words"])
df.collect()

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import DCT

df1 = spark.createDataFrame([(Vectors.dense([5.0, 8.0, 6.0]),)], ["vec"])
dct = DCT(inverse=False, inputCol="vec", outputCol="resultVec")
df2 = dct.transform(df1)
df2.show()

ranges = sc.parallelize([[12, 45], [9, 11], [31, 122], [88, 109], [17, 61]])
print type(ranges)

df = ranges.toDF(schema=['a', 'b'])
df.show()
print type(df)
print df.printSchema()

from pyspark.mllib.regression import LabeledPoint
import numpy as np

lp = sc.parallelize([LabeledPoint(1, np.array([1, 6, 7])), LabeledPoint(0, np.array([12, 2, 9]))])
lp.collect(), type(lp)

lp.toDF().show()

df = spark.createDataFrame([(1, Vectors.dense([7, 2, 9]), 'ignored'),
                            (0, Vectors.dense([6, 3, 1]), 'useless')], ["label", "features", "extra"])
df.show()

df.first()

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(df)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

from pyspark.ml.feature import HashingTF

tf = HashingTF(numFeatures=2**18, inputCol="words", outputCol="features")
df = spark.createDataFrame([(1, ['There', 'will', 'be', 'cake'],), (0, ['I', 'will', 'run', 'again'],)], ["label", "words"])
out = tf.transform(df)
out.show()

out.first()

ham = sc.textFile('ham.txt')
spam = sc.textFile('spam.txt')

hamLabelFeatures = ham.map(lambda email: [0, email.split()])
spamLabelFeatures = spam.map(lambda email: [1, email.split()])
trainRDD = hamLabelFeatures.union(spamLabelFeatures)

trainDF = trainRDD.toDF(schema=["label", "words"])
trainDF = tf.transform(trainDF)
trainDF.show()

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trainDF)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

test = spark.createDataFrame([(['Fox', 'and', 'two', 'are', 'two', 'things'],)], ["words"])
lrModel.transform(tf.transform(test)).select('prediction').show()

