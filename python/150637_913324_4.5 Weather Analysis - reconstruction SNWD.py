#setup
data_dir='../../Data/Weather'
file_index='SBBBSBSS'
m='SNWD'



# Enable automiatic reload of libraries
#%load_ext autoreload
#%autoreload 2 # means that all modules are reloaded before every command

#%matplotlib inline
get_ipython().run_line_magic('pylab', 'inline')
import numpy as np

import findspark
findspark.init()

import sys
sys.path.append('./lib')

from numpy_pack import packArray,unpackArray

from Eigen_decomp import Eigen_decomp
from YearPlotter import YearPlotter
from recon_plot import recon_plot

from import_modules import import_modules,modules
import_modules(modules)

from ipywidgets import interactive,widgets

from pyspark import SparkContext
#sc.stop()

sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStats.py','lib/recon_plot.py','lib/Eigen_decomp.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)




from pickle import load

#read statistics
print data_dir
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
measurements=STAT.keys()
print 'keys from STAT=',measurements

#read data
filename=data_dir+'/US_Weather_%s.parquet'%file_index
df_in=sqlContext.read.parquet(filename)
#filter in 
df=df_in.filter(df_in.measurement==m)
df.show(5)

import pylab as plt
import numpy as np
fig,axes=plt.subplots(2,1, sharex='col', sharey='row',figsize=(10,6));
k=4
EigVec=np.matrix(STAT[m]['eigvec'][:,:k])
Mean=STAT[m]['Mean']
YearPlotter().plot(Mean,fig,axes[0],label='Mean',title=m+' Mean')
YearPlotter().plot(EigVec,fig,axes[1],title=m+' Eigs',labels=['eig'+str(i+1) for i in range(k)])
                                             

v=[np.array(EigVec[:,i]).flatten() for i in range(np.shape(EigVec)[1])]

#  x=0 in the graphs below correspond to the fraction of the variance explained by the mean alone
#  x=1,2,3,... are the residuals for eig1, eig1+eig2, eig1+eig2+eig3 ...
fig,ax=plt.subplots(1,1);
eigvals=STAT[m]['eigval']; eigvals/=sum(eigvals); cumvar=np.cumsum(eigvals); cumvar=100*np.insert(cumvar,0,0)
ax.plot(cumvar[:10]); 
ax.grid(); 
ax.set_ylabel('Percent of variance explained')
ax.set_xlabel('number of eigenvectors')
ax.set_title('Percent of variance explained');

def decompose(row):
    """compute residual and coefficients for decomposition           

    :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
    :returns: the input row with additional information from the eigen-decomposition.
    :rtype: SparkSQL Row 

    Note that Decompose is designed to run inside a spark "map()" command.
    Mean and v are sent to the workers as local variables of "Decompose"

    """
    Series=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
    recon=Eigen_decomp(None,Series,Mean,v);
    total_var,residuals,reductions,coeff=recon.compute_var_explained()
    #print coeff
    residuals=[float(r) for r in residuals[1]]
    coeff=[float(r) for r in coeff[1]]
    D=row.asDict()
    D['total_var']=float(total_var[1])
    D['res_mean']=residuals[0]
    for i in range(1,len(residuals)):
        D['res_'+str(i)]=residuals[i]
        D['coeff_'+str(i)]=coeff[i-1]
    return Row(**D)

rdd2=df.rdd.map(decompose)
df2=sqlContext.createDataFrame(rdd2)
row,=df2.take(1)

#filter out vectors for which the mean is a worse approximation than zero.
print 'before filter',df2.count()
df3=df2.filter(df2.res_mean<1)
print 'after filter',df3.count()

# Sort entries by increasing values of ers_3
df3=df3.sort(df3.res_3,ascending=True)

def plot_decomp(row,Mean,v,fig=None,ax=None,Title=None,interactive=False,coeff_val=1):
    """Plot a single reconstruction with an informative title

    :param row: SparkSQL Row that contains the measurements for a particular station, year and measurement. 
    :param Mean: The mean vector of all measurements of a given type
    :param v: eigen-vectors for the distribution of measurements.
    :param fig: a matplotlib figure in which to place the plot
    :param ax: a matplotlib axis in which to place the plot
    :param Title: A plot title over-ride.
    :param interactive: A flag that indicates whether or not this is an interactive plot (widget-driven)
    :returns: a plotter returned by recon_plot initialization
    :rtype: recon_plot

    """
    target=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
    if Title is None:
        Title= 'coeff %s=%s, reconst. error(res %s)=%s'%(coeff_val,row['coeff_' + str(coeff_val)],coeff_val,row['res_' + str(coeff_val)])
    eigen_decomp=Eigen_decomp(range(1,366),target,Mean,v)
    plotter=recon_plot(eigen_decomp,year_axis=True,fig=fig,ax=ax,interactive=interactive,Title=Title)
    return plotter

def plot_recon_grid(rows,column_n=4, row_n=3, figsize=(15,10),coeff_val=1):
    """plot a grid of reconstruction plots

    :param rows: Data rows (as extracted from the measurements data-frame
    :param column_n: number of columns
    :param row_n:  number of rows
    :param figsize: Size of figure
    :returns: None
    :rtype: 

    """
    fig,axes=plt.subplots(row_n,column_n, sharex='col', sharey='row',figsize=figsize);
    k=0
    for i in range(row_n):
        for j in range(column_n):
            row=rows[k]
            k+=1
            #_title='%3.2f,r1=%3.2f,r2=%3.2f,r3=%3.2f'\
            #        %(row['res_mean'],row['res_1'],row['res_2'],row['res_3'])
            #print i,j,_title,axes[i,j]
            plot_decomp(row,Mean,v,fig=fig,ax=axes[i,j],interactive=False,coeff_val=coeff_val)
    return None

import gmplot
sqlContext.registerDataFrameAsTable(df,'weather')
Query="SELECT latitude,longitude FROM weather"
df1 = sqlContext.sql(Query)
lat_long = df1.collect()
latitude = [row[0] for row in lat_long]
longitude = [row[1] for row in lat_long]
gmap = gmplot.GoogleMapPlotter(latitude[0], longitude[0], 16)
gmap.scatter(latitude, longitude, 'b', marker=True)
gmap.heatmap(latitude, longitude)
gmap.draw("Sudarshan_map.html")

df4=df3.sort(df3.coeff_1)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=1)

df4=df3.sort(df3.coeff_1,ascending=False)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=1)

df4=df3.sort(df3.coeff_2)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=2)

df4=df3.sort(df3.coeff_2,ascending=False)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=2)

df4=df3.sort(df3.coeff_3)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=3)

df4=df3.sort(df3.coeff_3,ascending=False)
rows=df4.take(4)
plot_recon_grid(rows,column_n=2, row_n=2,coeff_val=3)

get_ipython().run_line_magic('pinfo', 'df3.sort')

df5=df3.filter(df3.res_2<0.4).sort(df3.coeff_2,ascending=False)
rows=df5.take(12)
df5.select('coeff_2','res_2').show(4)

plot_recon_grid(rows)

row=rows[0]
target=np.array(unpackArray(row.vector,np.float16),dtype=np.float64)
eigen_decomp=Eigen_decomp(None,target,Mean,v)
total_var,residuals,reductions,coeff=eigen_decomp.compute_var_explained()
res=residuals[1]
print 'residual normalized norm  after mean:',res[0]
print 'residual normalized norm  after mean + top eigs:',res[1:]

plotter=recon_plot(eigen_decomp,year_axis=True,interactive=True)
display(plotter.get_Interactive())

# A function for plotting the CDF of a given feature
def plot_CDF(feat):
    rows=df3.select(feat).sort(feat).collect()
    vals=[r[feat] for r in rows]
    P=np.arange(0,1,1./(len(vals)+1))
    vals=[vals[0]]+vals
    plot(vals,P)
    title('cumulative distribution of '+feat)
    ylabel('number of instances')
    xlabel(feat)
    grid()
    

plot_CDF('res_2')

plot_CDF('coeff_2')

# A function for plotting the CDF of a given feature
def plot_CDF(feat1,feat2,feat3,title_param):
    rows1=df3.select(feat1).sort(feat1).collect()
    vals1=[r[feat1] for r in rows1]
    P1=np.arange(0,1,1./(len(vals1)+1))
    vals1=[vals1[0]]+vals1
    rows2=df3.select(feat2).sort(feat2).collect()
    vals2=[r[feat2] for r in rows2]
    vals2=[vals2[0]]+vals2
    P2=np.arange(0,1,1./(len(vals2)))
    rows3=df3.select(feat3).sort(feat3).collect()
    vals3=[r[feat3] for r in rows3]
    vals3=[vals3[0]]+vals3
    P3=np.arange(0,1,1./(len(vals3)))
    plot(vals1,P1,label=feat1)
    plot(vals2,P2,label=feat2)
    plot(vals3,P3,label=feat3)
    title(title_param)
    ylabel('number of instances')
    xlabel('residual')
    legend()
    grid()
    
plot_CDF('res_1','res_2','res_3','cumulative distribution of residual 1,2 and 3')

plot_CDF('coeff_1','coeff_2','coeff_3','cumulative distribution of coefficients 1,2 and 3')

filename=data_dir+'/decon_'+file_index+'_'+m+'.parquet'
get_ipython().system('rm -rf $filename')
df3.write.parquet(filename)

get_ipython().system('du -sh $data_dir/*.parquet')



