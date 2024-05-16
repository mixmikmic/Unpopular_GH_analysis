get_ipython().system('pip install --upgrade --user pixiedust')

get_ipython().system('pip install --upgrade --user pixiedust-flightpredict')

import pixiedust_flightpredict
pixiedust_flightpredict.configure()

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from numpy import array
import numpy as np
import math
from datetime import datetime
from dateutil import parser
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
logRegModel = LogisticRegressionWithLBFGS.train(labeledTrainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , iterations=1000, validateData=False, intercept=False)
print(logRegModel)

from pyspark.mllib.classification import NaiveBayes
#NaiveBayes requires non negative features, set them to 0 for now
modelNaiveBayes = NaiveBayes.train(labeledTrainingData.map(lambda lp: LabeledPoint(lp.label,                     np.fromiter(map(lambda x: x if x>0.0 else 0.0,lp.features.toArray()),dtype=np.int)               ))          )

print(modelNaiveBayes)

from pyspark.mllib.tree import DecisionTree
modelDecisionTree = DecisionTree.trainClassifier(labeledTrainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , numClasses=training.getNumClasses(), categoricalFeaturesInfo={})
print(modelDecisionTree)

from pyspark.mllib.tree import RandomForest
modelRandomForest = RandomForest.trainClassifier(labeledTrainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , numClasses=training.getNumClasses(), categoricalFeaturesInfo={},numTrees=100)
print(modelRandomForest)

display(testData)

import pixiedust_flightpredict
from pixiedust_flightpredict import *
pixiedust_flightpredict.flightPredict("LAS")

import pixiedust_flightpredict
pixiedust_flightpredict.displayMapResults()

