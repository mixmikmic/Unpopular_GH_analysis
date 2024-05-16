import data
import pandas as pd

appliance = data.alldata.copy()
appliance.head()

import numpy as np
appliances = appliance.columns[:13]

appliance['Day'] = appliance.index.dayofyear
appliance['Hour'] = appliance.index.hour
appliance.head()

appliance_list_df = []
for i in range(len(appliances)):   
    a = appliance[[i,-1,-2]]
    a = a.resample('1h').mean()
    a = a.pivot(index='Day', columns='Hour')
    a = a.replace(np.inf,np.nan)
    a = a.fillna(0)
    appliance_list_df.append(a)

from sklearn.cluster import KMeans

appliance_inertias = []
for i in range(len(appliances)):
    two = KMeans(n_clusters=2,random_state=0).fit(appliance_list_df[i]).inertia_
    three = KMeans(n_clusters=3,random_state=0).fit(appliance_list_df[i]).inertia_
    a = [two, three]
    appliance_inertias.append(a)

appliance_inertias

#this will tell us if we should either go with two or three clusters for each appliance power consumption
clusters_for_min_inertia = []
for i in range(len(appliance_inertias)):
    clusters_for_min_inertia.append(appliance_inertias[i].index(min(appliance_inertias[i]))+2)
clusters_for_min_inertia 

import matplotlib.pyplot as plt

for l in range(len(appliances)):

    three_kmeans = KMeans(n_clusters=2,random_state=0).fit(appliance_list_df[l])

    hours = np.linspace(0,23,24)
    cluster_one = appliance_list_df[l].ix[three_kmeans.labels_ == 0]
    cluster_two = appliance_list_df[l].ix[three_kmeans.labels_ == 1]
    
    plt.figure(l)
    plt.figure(figsize=(6,4))

    plt.subplot(2,1,1)
    cluster_one = cluster_one.as_matrix(columns=None)
    for x in range(len(cluster_one)):
        plt.plot(hours,cluster_one[x],color='gray')
    plt.plot(hours,three_kmeans.cluster_centers_[0],linewidth=5,color='k')
    plt.title(appliances[l])
    plt.xlim(0,23)
    plt.ylabel('Energy Use, kWh')

    plt.subplot(2,1,2)
    cluster_two = cluster_two.as_matrix(columns=None)
    for x in range(len(cluster_two)):
        plt.plot(hours,cluster_two[x],color='gray')
    plt.plot(hours,three_kmeans.cluster_centers_[1],linewidth=5,color='k',)
    plt.xlim(0,23)
    plt.ylabel('Energy Use, kWh')
    plt.xlabel('Hour of Day')

