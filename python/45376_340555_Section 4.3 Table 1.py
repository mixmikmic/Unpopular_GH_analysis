import data
import pandas as pd

mydata = data.alldata.copy()
mydata

from sklearn import tree
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

mydata1 = mydata.copy()
x3 = mydata1[['television','fan','fridge','laptop computer','electric heating element','oven','unknown','washing machine','microwave','toaster','sockets','cooker']]
#xrange = np.arange(x3.min(),x3.max(),(x3.max()-x3.min())/100).reshape(100,1)
y1 = mydata1['Kitchen'].astype(float)
y2 = mydata1['LivingRoom'].astype(float)
y3 = mydata1['StoreRoom'].astype(float)
y4 = mydata1['Room1'].astype(float)
y5 = mydata1['Room2'].astype(float)




reg1 = tree.DecisionTreeClassifier(max_depth=10) 
reg1.fit(x3,y1)
reg1.score(x3,y1)

reg2 = tree.DecisionTreeClassifier(max_depth=10) 
reg2.fit(x3,y2)
reg2.score(x3,y2)

reg3 = tree.DecisionTreeClassifier(max_depth=10) 
reg3.fit(x3,y3)
reg3.score(x3,y3)

reg4 = tree.DecisionTreeClassifier(max_depth=10)
reg4.fit(x3,y4)
reg4.score(x3,y4)

reg5 = tree.DecisionTreeClassifier(max_depth=10)
reg5.fit(x3,y5)
reg3.score(x3,y5)

