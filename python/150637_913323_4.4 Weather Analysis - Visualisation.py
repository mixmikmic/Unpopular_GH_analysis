import pandas as pd
import numpy as np
import sklearn as sk
import urllib
import math
get_ipython().run_line_magic('pylab', 'inline')

import findspark
findspark.init()

from pyspark import SparkContext
#sc.stop()
sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/computeStats.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)

import sys
sys.path.append('./lib')

import numpy as np
from numpy_pack import packArray,unpackArray
#from spark_PCA import computeCov
from computeStats import computeOverAllDist, STAT_Descriptions

### Read the data frame from pickle file

data_dir='../../Data/Weather'
file_index='SBBBSBSS'

from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
print 'keys from STAT=',STAT.keys()

#read data
filename=data_dir+'/US_Weather_%s.parquet'%file_index

df=sqlContext.read.parquet(filename)
print df.count()
df.show(50)

sqlContext.registerDataFrameAsTable(df,'weather')
Query="SELECT * FROM weather\n\tWHERE measurement='%s' and station='%s'"%('TOBS','USC00242886')
print Query
df1 = sqlContext.sql(Query)
print df1.count(),'rows'
df1.show(2)
rows=df1.rdd.map(lambda row:unpackArray(row['vector'],np.float16)).collect()
T=np.vstack(rows)
T=T/10.  # scaling to make the temperature be in centingrates

sqlContext.registerDataFrameAsTable(df,'weather')
Query="SELECT count(year) as year_cnt,station FROM weather\n\tWHERE measurement='%s' GROUP BY station ORDER BY year_cnt DESC"%('TOBS') 
print Query
df1 = sqlContext.sql(Query)
print df1.count(),'rows'
df1.show(50)

from YearPlotter import YearPlotter
fig, ax = plt.subplots(figsize=(10,7));
YP=YearPlotter()
YP.plot(T[:2,:].transpose(),fig,ax,title='PRCP')
#title('A sample of graphs');

def plot_pair(pair,func):
    j=0
    fig,X=subplots(1,2,figsize=(16,6))
    axes=X.reshape(2)
    for m in pair:
        axis = axes[j]
        j+=1
        func(m,fig,axis)
        
def plot_valid(m,fig,axis):
    valid_m=STAT[m]['NE']
    YP.plot(valid_m,fig,axis,title='valid-counts '+m)
    

plot_pair(['TMIN','TMAX'],plot_valid)

plot_pair(['TOBS','PRCP'],plot_valid)

plot_pair(['SNOW', 'SNWD'],plot_valid)

def plot_mean_std(m,fig,axis):
    mean=STAT[m]['Mean'] / 10
    std=np.sqrt(STAT[m]['Var']) / 10
    graphs=np.vstack([mean-std,mean,mean+std]).transpose()
    YP.plot(graphs,fig,axis,title='Mean+-std   '+m)

b = plot_pair(['TMIN','TMAX'],plot_mean_std)

plot_pair(['TOBS','PRCP'],plot_mean_std)

plot_pair(['SNOW', 'SNWD'],plot_mean_std)

def plot_eigen(m,fig,axis):
    EV=STAT[m]['eigvec']
    YP.plot(EV[:,:3],fig,axis,title='Top Eigenvectors '+m)

plot_pair(['TMIN','TMAX'],plot_eigen)

plot_pair(['TOBS','PRCP'],plot_eigen)

plot_pair(['SNOW', 'SNWD'],plot_eigen)

def pltVarExplained(j):
    subplot(1,3,j)
    EV=STAT[m]['eigval']
    k=5
    plot(([0,]+list(cumsum(EV[:k])))/sum(EV))
    title('Percentage of Variance Explained for '+ m)
    ylabel('Percentage of Variance')
    xlabel('# Eigenvector')
    grid()
    

f=plt.figure(figsize=(15,4))
j=1
for m in ['TMIN', 'TOBS', 'TMAX']: #,
    pltVarExplained(j)
    j+=1

f=plt.figure(figsize=(15,4))
j=1
for m in ['SNOW', 'SNWD', 'PRCP']:
    pltVarExplained(j)
    j+=1 

from scipy import stats
def return_temp_data(measurement):

    Query_temp = "SELECT * FROM weather\n\tWHERE measurement='%s' and station='%s'"%('TOBS','USC00243013')
    print Query_temp
    df_T = sqlContext.sql(Query_temp)
    print df_T.count(),'rows'
    df_T.show(2)
    rows=df_T.rdd.map(lambda row:unpackArray(row['vector'],np.float16)).collect()
    T_max = np.vstack(rows)
    return T_max

def plot_winter(T,measurement):
    winter_temp = []
    for temp in T:
        winter_temp.append(np.nanmean(temp))
    xi = np.arange(0,len(winter_temp))
    slope, intercept, r_value, p_value, std_err = stats.linregress(xi,winter_temp)
    line = slope*xi+intercept
    print "slope is :"+str(slope)
    plt.plot(xi,winter_temp,'o', xi, line)
    plt.title("Mean "+measurement+" statistics for past "+str(len(xi))+" years")
    plt.ylabel(measurement)
    plt.xlabel("years")

T_min = return_temp_data('TOBS')
plot_winter(T_min,'TOBS')

T_min = return_temp_data('SNOW')
plot_winter(T_min,'SNOW')

#sc.stop()

