#import some libraries and tell ipython we want inline figures rather than interactive figures. 
import matplotlib.pyplot as plt, pandas as pd, numpy as np, matplotlib as mpl

from __future__ import print_function

get_ipython().magic('matplotlib inline')
pd.options.display.mpl_style = 'default' #load matplotlib for plotting
plt.style.use('ggplot') #im addicted to ggplot. so pretty.
mpl.rcParams['font.family'] = ['Bitstream Vera Sans']

rookie_df = pd.read_pickle('nba_bballref_rookie_stats_2016_Mar_15.pkl') #here's the rookie year data

rook_games = rookie_df['Career Games']>50
rook_year = rookie_df['Year']>1980

#remove rookies from before 1980 and who have played less than 50 games. I also remove some features that seem irrelevant or unfair
rookie_df_games = rookie_df[rook_games & rook_year] #only players with more than 50 games. 
rookie_df_drop = rookie_df_games.drop(['Year','Career Games','Name'],1)

from sklearn.preprocessing import StandardScaler

df = pd.read_pickle('nba_bballref_career_stats_2016_Mar_15.pkl')
df = df[df['G']>50]
df_drop = df.drop(['Year','Name','G','GS','MP','FG','FGA','FG%','3P','2P','FT','TRB','PTS','ORtg','DRtg','PER','TS%','3PAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','OWS','DWS','WS','WS/48','OBPM','DBPM','BPM','VORP'],1)
X = df_drop.as_matrix() #take data out of dataframe
ScaleModel = StandardScaler().fit(X)
X = ScaleModel.transform(X)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

reduced_model = PCA(n_components=5, whiten=True).fit(X)

reduced_data = reduced_model.transform(X) #transform data into the 5 PCA components space
final_fit = KMeans(n_clusters=6).fit(reduced_data) #fit 6 clusters
df['kmeans_label'] = final_fit.labels_ #label each data point with its clusters

import statsmodels.api as sm 
from sklearn.metrics import mean_absolute_error #import function for calculating mean squared error. 

X = rookie_df.as_matrix() #take data out of dataframe

cluster_labels = df[df['Year']>1980]['kmeans_label']
rookie_df_drop['kmeans_label'] = cluster_labels #label each data point with its clusters

plt.figure(figsize=(8,6));

estHold = [[],[],[],[],[],[]]

score = []

for i,group in enumerate(np.unique(final_fit.labels_)):
           
    Grouper = df['kmeans_label']==group #do one regression at a time
    Yearer = df['Year']>1980
    
    Group1 = df[Grouper & Yearer]
    Y = Group1['WS/48'] #get predictor data
    
    Group1_rookie = rookie_df_drop[rookie_df_drop['kmeans_label']==group]
    Group1_rookie = Group1_rookie.drop(['kmeans_label'],1) #get predicted data

    X = Group1_rookie.as_matrix() #take data out of dataframe    
    
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    est = sm.OLS(Y,X) #fit with linear regression model
    est = est.fit()
    estHold[i] = est
    score.append(mean_absolute_error(Y,est.predict(X))) #calculate the mean squared error
    #print est.summary()
    
    plt.subplot(3,2,i+1) #plot each regression's prediction against actual data
    plt.plot(est.predict(X),Y,'o',color=plt.rcParams['axes.color_cycle'][i])
    plt.plot(np.arange(-0.1,0.25,0.01),np.arange(-0.1,0.25,0.01),'-')
    plt.title('Group %d'%(i+1))
    plt.text(0.15,-0.05,'$r^2$=%.2f'%est.rsquared)
    plt.xticks([0.0,0.12,0.25])
    plt.yticks([0.0,0.12,0.25]); 

from sklearn.linear_model import LinearRegression #I am using sklearns linear regression because it plays well with their cross validation function
from sklearn import cross_validation #import the cross validation function

X = rookie_df.as_matrix() #take data out of dataframe

cluster_labels = df[df['Year']>1980]['kmeans_label']
rookie_df_drop['kmeans_label'] = cluster_labels #label each data point with its clusters

for i,group in enumerate(np.unique(final_fit.labels_)):
           
    Grouper = df['kmeans_label']==group #do one regression at a time
    Yearer = df['Year']>1980
    
    Group1 = df[Grouper & Yearer]
    Y = Group1['WS/48'] #get predictor data
    
    Group1_rookie = rookie_df_drop[rookie_df_drop['kmeans_label']==group]
    Group1_rookie = Group1_rookie.drop(['kmeans_label'],1) #get predicted data

    X = Group1_rookie.as_matrix() #take data out of dataframe    
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    
    est = LinearRegression() #fit with linear regression model
    
    this_scores = cross_validation.cross_val_score(est, X, Y,cv=10, scoring='mean_absolute_error') #find mean square error across different datasets via cross validations
    print('Group '+str(i))
    print('Initial Mean Absolute Error: '+str(score[i])[0:6])
    print('Cross Validation MAE: '+str(np.median(np.abs(this_scores)))[0:6]) #find the mean MSE across validations

#plot the residuals. there's obviously a problem with under/over prediction

plt.figure(figsize=(8,6));

for i,group in enumerate(np.unique(final_fit.labels_)):
           
    Grouper = df['kmeans_label']==group #do one regression at a time
    Yearer = df['Year']>1980
    
    Group1 = df[Grouper & Yearer]
    Y = Group1['WS/48'] #get predictor data
    resid = estHold[i].resid #extract residuals
        
    plt.subplot(3,2,i+1) #plot each regression's prediction against actual data
    plt.plot(Y,resid,'o',color=plt.rcParams['axes.color_cycle'][i])
    plt.title('Group %d'%(i+1))
    plt.xticks([0.0,0.12,0.25])
    plt.yticks([-0.1,0.0,0.1]); 

