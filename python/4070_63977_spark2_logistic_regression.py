from pyspark.ml.classification import LogisticRegression

training = spark.read.format("libsvm").load("/Users/jhalverson/software/spark-2.0.0-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")
print type(training)
print training.first()

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

lrModel = lr.fit(training)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

