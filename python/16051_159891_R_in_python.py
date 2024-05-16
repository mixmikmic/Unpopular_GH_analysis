#first import the libraries I always use. 
import numpy as np, scipy.stats, pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().magic('matplotlib inline')
pd.options.display.mpl_style = 'default'
plt.style.use('ggplot')
mpl.rcParams['font.family'] = ['Bitstream Vera Sans']

import random

random.seed(1) #seed random number generator
cond_1 = [random.gauss(600,30) for x in range(30)] #condition 1 has a mean of 600 and standard deviation of 30
cond_2 = [random.gauss(650,30) for x in range(30)] #u=650 and sd=30
cond_3 = [random.gauss(600,30) for x in range(30)] #u=600 and sd=30

plt.bar(np.arange(1,4),[np.mean(cond_1),np.mean(cond_2),np.mean(cond_3)],align='center') #plot data
plt.xticks([1,2,3]);

get_ipython().magic('load_ext rpy2.ipython')

#pop the data into R
get_ipython().magic('Rpush cond_1 cond_2 cond_3')

#label the conditions
get_ipython().magic("R Factor <- c('Cond1','Cond2','Cond3')")
#create a vector of conditions
get_ipython().magic('R idata <- data.frame(Factor)')

#combine data into single matrix
get_ipython().magic('R Bind <- cbind(cond_1,cond_2,cond_3)')
#generate linear model
get_ipython().magic('R model <- lm(Bind~1)')

#load the car library. note this library must be installed.
get_ipython().magic('R library(car)')
#run anova
get_ipython().magic('R analysis <- Anova(model,idata=idata,idesign=~Factor,type="III")')
#create anova summary table
get_ipython().magic('R anova_sum = summary(analysis)')

#move the data from R to python
get_ipython().magic('Rpull anova_sum')
print anova_sum

random.seed(1)

cond_1a = [random.gauss(600,30) for x in range(30)] #u=600,sd=30
cond_2a = [random.gauss(650,30) for x in range(30)] #u=650,sd=30
cond_3a = [random.gauss(600,30) for x in range(30)] #u=600,sd=30

cond_1b = [random.gauss(600,30) for x in range(30)] #u=600,sd=30
cond_2b = [random.gauss(550,30) for x in range(30)] #u=550,sd=30
cond_3b = [random.gauss(650,30) for x in range(30)] #u=650,sd=30

width = 0.25
plt.bar(np.arange(1,4)-width,[np.mean(cond_1a),np.mean(cond_2a),np.mean(cond_3a)],width)
plt.bar(np.arange(1,4),[np.mean(cond_1b),np.mean(cond_2b),np.mean(cond_3b)],width,color=plt.rcParams['axes.color_cycle'][0])
plt.legend(['A','B'],loc=4)
plt.xticks([1,2,3]);

get_ipython().magic('Rpush cond_1a cond_1b cond_2a cond_2b cond_3a cond_3b')

get_ipython().magic("R Factor1 <- c('A','A','A','B','B','B')")
get_ipython().magic("R Factor2 <- c('Cond1','Cond2','Cond3','Cond1','Cond2','Cond3')")
get_ipython().magic('R idata <- data.frame(Factor1, Factor2)')

#make sure the vectors appear in the same order as they appear in the dataframe
get_ipython().magic('R Bind <- cbind(cond_1a, cond_2a, cond_3a, cond_1b, cond_2b, cond_3b)')
get_ipython().magic('R model <- lm(Bind~1)')

get_ipython().magic('R library(car)')
get_ipython().magic('R analysis <- Anova(model, idata=idata, idesign=~Factor1*Factor2, type="III")')
get_ipython().magic('R anova_sum = summary(analysis)')
get_ipython().magic('Rpull anova_sum')

print anova_sum

