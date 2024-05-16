from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

spam = sc.textFile("spam.txt")
ham = sc.textFile("ham.txt")

print spam.count(), spam.first()
print ham.count(), ham.first()

htf = HashingTF(numFeatures=10000)

spamVecs = spam.map(lambda line: htf.transform(line.split()))
hamVecs = ham.map(lambda line: htf.transform(line.split()))

spamTrain = spamVecs.map(lambda vctr: LabeledPoint(1, vctr))
hamTrain = hamVecs.map(lambda vctr: LabeledPoint(0, vctr))

training = spamTrain.union(hamTrain)
training.cache()

training.first()

lr = LogisticRegressionWithLBFGS()
lrModel = lr.train(training)

lrModel.predict(htf.transform('There was a candle on the cabinet.'.split()))

