import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import copy
import sklearn
from sklearn import tree, feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
get_ipython().magic('matplotlib inline')

f = open('/Users/apple/Desktop/data driven/project/dataset/oil consumption time series.csv')
t = pd.read_csv(f,sep=',', header='infer', parse_dates=[1])

t = t.set_index('Year')
del t.index.name

# Here is a part of example of the dataset
t

t.index.tolist()

fig = plt.figure(figsize=(10,5))

plt.plot(t['Total Consumption'],label='Total Consumption')
plt.plot(t['Northeast'],'r',label='Northeast Consumption')
plt.plot(t['Midwest'],'g',label='Midwest Consumption')
plt.plot(t['South'],'black',label='South Consumption')
plt.plot(t['West'],'purple',label='West Consumption')

# plt.title('Oil Consumption Census by Region and Division,1980-2001')
plt.xlabel('Year')
plt.ylabel('Oil Consumption (trillion Btu)')
            
fig.tight_layout()
plt.legend()
plt.show()

Consumption=[]
for i in range (9):
    val=float(t['Total Consumption'].values[i])
    Consumption.append(val)

Northeast=[]
for i in range (9):
    val=float(t['Northeast'].values[i])
    Northeast.append(val)
    
Midwest=[]
for i in range (9):
    val=float(t['Midwest'].values[i])
    Midwest.append(val)
    
South=[]
for i in range (9):
    val=float(t['South'].values[i])
    South.append(val)
    
West=[]
for i in range (9):
    val=float(t['West'].values[i])
    West.append(val)

fig = plt.figure(figsize=(10,5))
x = np.array(t.index.tolist())
width = 0.18
opacity = 0.6
plt.bar(x, Consumption, width, alpha=opacity,color="blue",label='Total Consumption')
plt.bar(x+width, Northeast, width, alpha=opacity,color="red",label='Northeast Consumption')
plt.bar(x+2*width, Midwest, width, alpha=opacity,color="green",label='Midwest Consumption')
plt.bar(x+3*width, South, width, alpha=opacity,color="black",label='South Consumption')
plt.bar(x+4*width, West, width, alpha=opacity,color="purple",label='West Consumption')

plt.xticks(x + 2*width, (t.index))

# plt.title('Oil Consumption Census by Region and Division,1980-2001')
plt.xlabel('Year')
plt.ylabel('Oil Consumption (trillion Btu)')
fig.tight_layout()
plt.legend()

y=Consumption

Year=t.index.tolist()

Households=[]
for i in range (9):
    val=float(t['Total Households'].values[i])
    Households.append(val)

Oil_price=[]
for i in range (9):
    val=float(t['Oil Price'].values[i])
    Oil_price.append(val)

Buildings=[]
for i in range (9):
    val=float(t['Total Residential Buildings'].values[i])
    Buildings.append(val)

Floorspace=[]
for i in range (9):
    val=float(t['Total Floorspace'].values[i])
    Floorspace.append(val)

# regression
X= []
for i in range(len(y)):
    tmp = []
    tmp.append(Year[i])
    tmp.append(Households[i])
    tmp.append(Oil_price[i])
    tmp.append(Buildings[i])
    tmp.append(Floorspace[i])
    X.append(tmp)

# fit a regression tree
clf = tree.DecisionTreeRegressor(max_depth=1)
clf = clf.fit(X,y)

# Using the model above to predict one instance
result=clf.predict([2003,11,41,9,26])[0]
print('The predicted oil consumption for 2003 is '+str(result)+' million btu.')

#  calculate the score
clf.score(X,y)

clf.feature_importances_

sfm = SelectFromModel(clf, threshold='median')
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
sfm.transform(X)

# change the parameter to optimize the regression model
clf2 = tree.DecisionTreeRegressor(max_depth=3)
clf2 = clf2.fit(X,y)
clf2.score(X,y)

clf2.feature_importances_

sfm = SelectFromModel(clf2, threshold='median')
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
sfm.transform(X)

clf4 = tree.DecisionTreeRegressor(max_depth=4)
clf4 = clf4.fit(X,y)
clf4.score(X,y)

# Using the model above to predict one instance
result=clf4.predict([2003,11,41,9,26])[0]
print('The predicted oil consumption for 2003 is '+str(result)+' million btu.')

#feature_importances_  
# The feature importances. The higher, the more important the feature.

clf4.feature_importances_ 

sfm = SelectFromModel(clf4, threshold='median')
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
sfm.transform(X)

# get rid of the Building 
X2= []
for i in range(len(y)):
    tmp = []
    tmp.append(Year[i])
    tmp.append(Households[i])
    tmp.append(Oil_price[i])
#     tmp.append(Buildings[i])
    tmp.append(Floorspace[i])
    X2.append(tmp)

clf2_ = tree.DecisionTreeRegressor(max_depth=3)
clf2_ = clf2_.fit(X2,y)
clf2_.score(X2,y)

clf2_.feature_importances_

x1=pd.read_excel('/Users/apple/Desktop/data driven/project/dataset/oil4.xls')
explode = (0.05, 0.1, 0.1)
labels = x1.columns
x1=np.array(x1)
plt.pie(x1[0],explode=explode,labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

oil=pd.read_excel('/Users/apple/Desktop/data driven/project/dataset/oil.xls')
oil=oil.drop(oil.columns[[1]],axis=1)
oil=oil.drop(oil.columns[[1]],axis=1)
oil=oil.drop(oil.columns[[0]],axis=1)
oil=oil.dropna()
oil=oil.T
oilindex=oil.index
A=np.array(oil.dropna())
kmeans = KMeans(n_clusters=4, random_state=0).fit(A)
c=kmeans.labels_
a=kmeans.cluster_centers_
plt.figure(figsize=(16,9))
k1=0
k2=0
for i in range(len(A)):
    if c[i]==0:
        k1=k1+1
        plt.subplot(2,2,1)
        plt.plot(A[i],linewidth=0.5,color='grey')
        plt.text(6.3,87-k1*4,oilindex[i],fontsize=10)
    if c[i]==1:
        k2=k2+1
        plt.subplot(2,2,2)
        plt.plot(A[i],linewidth=0.5,color='grey')
        plt.text(6.2,190-k2*6,oilindex[i],fontsize=10)
    if c[i]==2:
        k1=k1+1
        plt.subplot(2,2,3)
        plt.plot(A[i],linewidth=0.5,color='grey')
        plt.text(5.8,120-k1*6,oilindex[i],fontsize=10)
    if c[i]==3:
        k2=k2+1
        plt.subplot(2,2,4)
        plt.plot(A[i],linewidth=0.5,color='grey')
        plt.text(5.8,82-k2*6,oilindex[i],fontsize=10)
text=oil.columns
for i in range(4):
    if i==0:
        plt.subplot(2,2,1)
        plt.plot(a[i],linewidth=2,color='black')
        plt.xlabel('Year''\n'
                   '(a)')
        plt.ylabel('Consumption per building (million Btu)')
#         plt.title('Census Region and Division(k=4) i=1')
        for j in range(len(text)):
            plt.text(j, 40, text[j], fontsize=10,rotation=55)
    if i==1:
        plt.subplot(2,2,2)
        plt.plot(a[i],linewidth=2,color='black')
        plt.xlabel('Year''\n'
                   '(b)')
        plt.ylabel('Consumption per building (million Btu)')
#         plt.title('Census Region and Division(k=4) i=2')
        for j in range(len(text)):
            plt.text(j, 140, text[j], fontsize=10,rotation=55)
    if i==2:
        plt.subplot(2,2,3)
        plt.plot(a[i],linewidth=2,color='black')
        plt.xlabel('Year''\n'
                   '(c)')
        plt.ylabel('Consumption per building (million Btu)')
#         plt.title('Census Region and Division(k=4) i=3')
        for j in range(len(text)):
            plt.text(j, 60, text[j], fontsize=10,rotation=55)
    if i==3:
        plt.subplot(2,2,4)
        plt.plot(a[i],linewidth=2,color='black')
        plt.xlabel('Year''\n'
                   '(d)')
        plt.ylabel('Consumption per building (million Btu)')
#         plt.title('Census Region and Division(k=4) i=4')
        for j in range(len(text)):
            plt.text(j, 20, text[j], fontsize=10,rotation=55)

# A1=pd.DataFrame(A)
# # A1=A1.T
# A1.boxplot()
# for j in range(len(text)):
#         plt.text(j, 280, text[j], fontsize=10,rotation=-60)
# plt.ylim([0,300])
# plt.xlabel('Region')
# plt.ylabel('Consumption per building (million Btu)')
# # plt.title('Census Region and Division')


get_ipython().magic('matplotlib inline')
oil=pd.read_excel('/Users/apple/Desktop/data driven/project/dataset/oil.xls')
oil=oil.drop(oil.columns[[1]],axis=1)
oil=oil.drop(oil.columns[[1]],axis=1)
oilindex=oil.index
A=np.array(oil.dropna())
A1=pd.DataFrame(A)
# A1=A1.T
A1.boxplot()
text=oil.columns
for j in range(len(text)):
        plt.text(j, 280, text[j], fontsize=10,rotation=-60)
plt.ylim([0,300])
plt.xlabel('Region')
plt.ylabel('Consumption per building (million Btu)')
# plt.title('Census Region and Division')

plt.imshow(A)
plt.colorbar()
for j in range(len(text)):
        plt.text(j, 10.5, text[j], fontsize=10,rotation=-60)
for j in range(len(oilindex)):
        plt.text(-3.2, j, oilindex[j], fontsize=10)
plt.xlabel('Region')
plt.ylabel('year')
# plt.title('Census Region and Division')



