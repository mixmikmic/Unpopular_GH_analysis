from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

ham = sc.textFile('ham.txt')
spam = sc.textFile('spam.txt')

print ham.count()
print ham.first()

print spam.count()
print spam.first()

tf = HashingTF(numFeatures=10000)
hamFeatures = ham.map(lambda email: tf.transform(email.split()))
spamFeatures = spam.map(lambda email: tf.transform(email.split()))

hamFeatures.first()

positiveClass = spamFeatures.map(lambda record: LabeledPoint(1, record))
negativeClass = hamFeatures.map(lambda record: LabeledPoint(0, record))

positiveClass.first()

trainingData = positiveClass.union(negativeClass).cache()
model = LogisticRegressionWithLBFGS.train(trainingData)

[model.predict(item) for item in hamFeatures.collect()]

[model.predict(item) for item in spamFeatures.collect()]

model.predict(tf.transform("Get a free mansion by sending 1 million dollars to me.".split()))

model.predict(tf.transform("Hi Mark, Let's meet at the coffee shop at 3 pm.".split()))

