import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set_context('poster')

file='data/rucio_transfer-events-2017.08.06.csv'
data = pd.read_csv(file)
data = data.drop('Unnamed: 0', axis=1)
# data=data.set_index(['submitted_at'])
print(data.head(5), '\n --------------------- \n')
data.info()

plt.plot(data['duration'] / 60, 'g')
plt.plot(data['prediction']/ 60, 'y')
# data['duration'].plot()
# data['prediction'].plot()
plt.show()

errors= data['duration'] - data['prediction']
print(errors.shape)
errors=np.reshape(errors, [errors.shape[0],1])
print(errors.shape)
plt.plot(errors, 'or')

plt.show()

errs = errors/60
print(np.max(errs))

bins=15
# bins=[-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60]
arr= plt.hist(errs, bins=bins)
for i in range(bins):
    plt.text(arr[1][i],arr[0][i],str(int(arr[0][i])))
plt.show()

arr2 = plt.hist(np.absolute(errs), bins=bins)
for i in range(bins):
    plt.text(arr2[1][i],arr2[0][i],str(int(arr2[0][i])))
plt.show()


i_1=0
i_2=0
i_3 = 0
i_4 = 0
i_5 = 0
i_10 = 0
i_20 = 0
i_30=0
i_40 =0
i_50 = 0
i_60=0
j=0
k=0
for err in errors:
    if err<=0:
        k+=1
    if np.absolute(err/ 60)<=1:
        i+=1
    if np.absolute(err/ 60)<=2:
        i_2+=1
    if np.absolute(err/ 60)<=3:
        i_3+=1
    if np.absolute(err/ 60)<=4:
        i_4+=1
    if np.absolute(err/ 60)<=5:
        i_5+=1
    if np.absolute(err/ 60)<=10:
        i_10+=1
    if np.absolute(err/ 60)<=20:
        i_20+=1
    if np.absolute(err/ 60)<=30:
        i_30+=1
    if np.absolute(err/ 60)<=40:
        i_40+=1
    if np.absolute(err/ 60)<=50:
        i_50+=1
    if np.absolute(err/ 60)<=60:
        i_60+=1
    else:
        j+=1
print('total values with error less than 1 minutes : {}  percentage :{} %'.format(i, (i/len(errors) *100)))
print('total values with error less than 2 minutes : {}  percentage :{} %'.format(i_2, (i_2/len(errors) *100)))
print('total values with error less than 3 minutes : {}  percentage :{} %'.format(i_3, (i_3/len(errors) *100)))
print('total values with error less than 5 minutes : {}  percentage :{} %'.format(i_5, (i_5/len(errors) *100)))
print('total values with error less than 10 minutes : {}  percentage :{} %'.format(i_10, (i_10/len(errors) *100)))
print('total values with error less than 30 minutes : {}  percentage :{} %'.format(i_30, (i_30/len(errors) *100)))
print('total values with error more than an hour : {}  percentage :{} %'.format(j, (j/len(errors) *100)))
print('total values with negative errors i.e transfers faster tha predicted by the model(positive anomalies) : {}  percentage :{} %'.format(k, (k/len(errors) *100)))
max_err = np.max(np.absolute(errors))
print('max error :{}  minutes'.format(max_err/60))

# sns.barplot(x='submitted_at', y=err_min, data=data)

cond = data['label']=='anomaly'
anomalies= data[cond]
normal_data = data[cond!=True]
assert len(normal_data)+len(anomalies)==len(data)

fig = plt.figure()
ax = fig.add_subplot(1, 1,1)
# x_norm = []
# y_norm = []
# x_anom = []
# y_anom = []
ax.scatter(normal_data['duration'], normal_data['prediction'],c='green', s=200.0, label='normal', alpha=0.3, edgecolors='none')
ax.scatter(anomalies['duration'], anomalies['prediction'],c='red', s=200.0, label='anomaly', alpha=0.3, edgecolors='none')
ax.plot(normal_data['duration'],normal_data['duration'], 'b', label='ideal')
# ax.plot(data['duration'], data['duration'], 'y', label='reality')
ax.legend()
plt.xlabel('duration', fontsize=16)
plt.ylabel('prediction', fontsize=16)

plt.scatter(normal_data['duration'], normal_data['prediction'], c='green')

plt.scatter(anomalies['duration'], anomalies['prediction'], c='red')

anomalies.shape

normal_data.shape

print('% of anomalies from {} events = {:.3f} % ; ({}) '.format(data.shape[0], (anomalies.shape[0]/data.shape[0])*100,anomalies.shape[0]))

anomalies.info()

delta= anomalies['duration']-anomalies['prediction']
plt.plot(delta, 'ro')
plt.show()

size_gb=anomalies['bytes']/1073741824
# print(size_gb.value_counts())

count, division = np.histogram(size_gb, bins = range(0,60))
size_gb.hist(bins=division)
count,division

c

data.info()

threshold=600

data['err'] = data['duration']-data['prediction']

def f(x):
    if x<=threshold:
        return 'normal'
    else:
        return 'anomaly'

data['correct_label']= data['err'].apply(lambda x : f(x))
data=data.drop('err', axis=1)
data.head()

cond = data['correct_label']=='anomaly'
anomalies= data[cond]
normal_data = data[cond!=True]
assert len(normal_data)+len(anomalies)==len(data)

len(anomalies)

delta= anomalies['duration']-anomalies['prediction']
delta= delta/60
plt.plot(delta, 'ro')
plt.xlabel('#events')
plt.ylabel('duration errors in minutes')
plt.show()

plt.scatter(anomalies['bytes'], anomalies['prediction'], c='yellow', alpha=0.3,  s=100, edgecolors=None, label='predicted')
plt.scatter(anomalies['bytes'], anomalies['duration'], c='green',alpha=0.3, s=100, edgecolors=None, label='actual')
plt.xlabel('bytes')
plt.ylabel('transfer duration in seconds')
plt.title('Duration vs Filesizes')
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(normal_data['duration'], normal_data['prediction'],c='green', s=200.0, label='normal', alpha=0.3, edgecolors='none')
ax.scatter(anomalies['duration'], anomalies['prediction'],c='red', s=200.0, label='anomaly', alpha=0.3, edgecolors='none')
ax.plot(normal_data['duration'],normal_data['duration'], 'b', label='ideal')
# ax.plot(data['duration'], data['duration'], 'y', label='reality')
ax.legend()
plt.xlabel('duration', fontsize=16)
plt.ylabel('prediction', fontsize=16)

size_gb=anomalies['bytes']/1073741824
# print(size_gb.value_counts())

count, division = np.histogram(size_gb, bins = range(0,60))
size_gb.hist(bins=division)
count,division

c= data['bytes']>=10*1073741824
v= data[c]
v.head()

v.shape

v

data['activity'].unique()

a=data['activity']=='Data Rebalancing'
a=data[a]

duration_minutes=data['duration']/60
bins=range(0,int(np.max(duration_minutes)), 5)
count, division = np.histogram(duration_minutes, bins = range(0,int(np.max(duration_minutes)), 5))
duration_minutes.hist(bins=division)
print(count,division)
for i in range(0,len(bins)-1):
    plt.text(division[i],count[i]+150000,str(int(count[i])), rotation=90)
plt.show()

duration_minutes=data['prediction']/60
bins=range(0,int(np.max(duration_minutes)), 5)
count, division = np.histogram(duration_minutes, bins = range(0,int(np.max(duration_minutes)), 5))
duration_minutes.hist(bins=division)
print(count,division)
for i in range(0,len(bins)-1):
    plt.text(division[i],count[i]+150000,str(int(count[i])), rotation=90)
    



