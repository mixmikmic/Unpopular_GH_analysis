get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv("2008.csv")
df.shape[0]

df = df[0:1000000]

df.columns

df[0:5]

df = pd.concat([df, pd.get_dummies(df["Origin"], prefix="Origin")], axis=1);
df = pd.concat([df, pd.get_dummies(df["Dest"  ], prefix="Dest"  )], axis=1);
df = df.dropna(subset=["ArrDelay"]) 
df["IsArrDelayed" ] = (df["ArrDelay"]>0).astype(int)
df[0:5]

train = df.sample(frac=0.8)
test  = df.drop(train.index)

#get the list of one hot encoding columns
OriginFeatCols = [col for col in df.columns if ("Origin_" in col)]
DestFeatCols   = [col for col in df.columns if ("Dest_"   in col)]
features = train[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
labels   = train["IsArrDelayed"]
featuresMatrix = features.as_matrix()
labelsMatrix   = labels  .as_matrix().reshape(-1,1)

featureSize     = features.shape[1]
labelSize       = 1
training_epochs = 25
batch_size      = 2500

from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.regularizers import l2, activity_l2
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

#DEFINE A CUSTOM CALLBACK
class IntervalEvaluation(Callback):
    def __init__(self): super(Callback, self).__init__()
    def on_epoch_end(self, epoch, logs={}): print("interval evaluation - epoch: %03d - loss:%8.6f" % (epoch, logs['loss']))

#DEFINE AN EARLY STOPPING FOR THE MODEL
earlyStopping = EarlyStopping(monitor='loss', patience=1, verbose=0, mode='auto')
        
#DEFINE THE MODEL
model = Sequential() 
model.add(Dense(labelSize, input_dim=featureSize, activation='sigmoid', W_regularizer=l2(1e-5))) 
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']) 

#FIT THE MODEL
model.fit(featuresMatrix, labelsMatrix, batch_size=batch_size, nb_epoch=training_epochs,verbose=0,callbacks=[IntervalEvaluation(),earlyStopping]);

coef = pd.DataFrame(data=model.layers[0].get_weights()[0], index=features.columns, columns=["Coef"])
coef = coef.reindex( coef["Coef"].abs().sort_values(axis=0,ascending=False).index )  #order by absolute coefficient magnitude
coef[ coef["Coef"].abs()>0 ] #keep only non-null coefficients
coef[ 0:10 ] #keep only the 10 most important coefficients

testFeature = test[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
pred = model.predict( testFeature.as_matrix() )
test["IsArrDelayedPred"] = pred
test[0:10]

fpr, tpr, _ = roc_curve(test["IsArrDelayed"], test["IsArrDelayedPred"])
AUC = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.3f)' % AUC)
plt.legend(loc=4)

AUC



