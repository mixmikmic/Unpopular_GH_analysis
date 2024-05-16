get_ipython().system('pip show flightPredict')

get_ipython().system('pip uninstall --yes flightPredict')

get_ipython().system('pip install --user --exists-action=w --egg git+https://github.com/ibm-watson-data-lab/simple-data-pipe-connector-flightstats.git#egg=flightPredict')

get_ipython().magic('matplotlib inline')
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from numpy import array
import numpy as np
import math
from datetime import datetime
from dateutil import parser

import flightPredict
sqlContext=SQLContext(sc)
flightPredict.sqlContext = sqlContext
flightPredict.cloudantHost='XXXX'
flightPredict.cloudantUserName='XXXX'
flightPredict.cloudantPassword='XXXX'
flightPredict.weatherUrl='https://XXXX:XXXXb@twcservice.mybluemix.net'

dbName = "flightstats_training_data_for_flight_delay_predictive_app_mega_set"
cloudantdata = flightPredict.loadDataSet(dbName,"training")
cloudantdata.printSchema()
cloudantdata.count()

flightPredict.scatterPlotForFeatures(cloudantdata,      "departureWeather.temp","arrivalWeather.temp","Departure Airport Temp", "Arrival Airport Temp")

flightPredict.scatterPlotForFeatures(cloudantdata,     "departureWeather.pressure","arrivalWeather.pressure","Departure Airport Pressure", "Arrival Airport Pressure")

flightPredict.scatterPlotForFeatures(cloudantdata, "departureWeather.wspd","arrivalWeather.wspd","Departure Airport Wind Speed", "Arrival Airport Wind Speed")

computeClassification = (lambda deltaDeparture: 0 if deltaDeparture<13 else (1 if deltaDeparture < 41 else 2))
def customFeatureHandler(s):
    if(s==None):
        return ["departureTime"]
    dt=parser.parse(s.departureTime)
    features=[]
    for i in range(0,7):
        features.append(1 if dt.weekday()==i else 0)
    return features
computeClassification=None
customFeatureHandler=None
numClasses = 5

trainingData = flightPredict.loadLabeledDataRDD("training", computeClassification, customFeatureHandler)
trainingData.take(5)

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
logRegModel = LogisticRegressionWithLBFGS.train(trainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , iterations=100, validateData=False, intercept=True)
print(logRegModel)

from pyspark.mllib.classification import NaiveBayes
#NaiveBayes requires non negative features, set them to 0 for now
modelNaiveBayes = NaiveBayes.train(trainingData.map(lambda lp: LabeledPoint(lp.label,                     np.fromiter(map(lambda x: x if x>0.0 else 0.0,lp.features.toArray()),dtype=np.int)               ))          )

print(modelNaiveBayes)

from pyspark.mllib.tree import DecisionTree
modelDecisionTree = DecisionTree.trainClassifier(trainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , numClasses=numClasses, categoricalFeaturesInfo={})
print(modelDecisionTree)

from pyspark.mllib.tree import RandomForest
modelRandomForest = RandomForest.trainClassifier(trainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , numClasses=numClasses, categoricalFeaturesInfo={},numTrees=100)
print(modelRandomForest)

dbTestName="flightstats_test_data_for_flight_delay_predictive_app"
testCloudantdata = flightPredict.loadDataSet(dbTestName,"test")
testCloudantdata.count()

testData = flightPredict.loadLabeledDataRDD("test",computeClassification, customFeatureHandler)
flightPredict.displayConfusionTable=True
flightPredict.runMetrics(trainingData,modelNaiveBayes,modelDecisionTree,logRegModel,modelRandomForest)

from flightPredict import run
run.useModels(modelDecisionTree,modelRandomForest)
run.runModel('BOS', "2016-02-08 20:15-0500", 'LAX', "2016-01-08 22:30-0500" )

rdd = sqlContext.sql("select deltaDeparture from training").map(lambda s: s.deltaDeparture)    .filter(lambda s: s < 50 and s > 12)
    
print(rdd.count())

histo = rdd.histogram(50)
    
#print(histo[0])
#print(histo[1])
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
bins = [i for i in histo[0]]

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Number of records')
plt.xlabel('Bin')
plt.title('Histogram')
intervals = [abs(j-i) for i,j in zip(bins[:-1], bins[1:])]
values=[sum(intervals[:i]) for i in range(0,len(intervals))]
plt.bar(values, histo[1], intervals, color='b', label = "Bins")
plt.xticks(bins[:-1],[int(i) for i in bins[:-1]])
plt.legend()

plt.show()



