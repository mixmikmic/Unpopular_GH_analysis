# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])

dimensions = ['weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 
    'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Sabotage Equipment', 
    'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)']

columns = dfs.columns

for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)

print("Loading DB...")
dfs_names = pd.read_csv("final_names.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
print(type(dfs_names['group']))

import collections
group_dict = collections.OrderedDict()
for name in dfs_names['group']:
    group_dict[(name,-1)] = []
    for index in range(len(dimensions)):
        group_dict[(name,-1)].append(0)
print(group_dict)

for index, gname in enumerate(dfs['gname']):
    for i in range(len(dimensions)):
        #if dimensions[i] == 'nkill':
        #   print(str(dfs[dimensions[i]][index]))
        if str(dfs[dimensions[i]][index]) == 'nan' or dfs[dimensions[i]][index] < 0 :
            continue
        group_dict[(gname,-1)][i] += dfs[dimensions[i]][index] 

def get_index(groups, name):
    for i in range(len(groups)):
        if groups[i] == name:
            return i
    return -1

unique_groups, group_counts = np.unique(dfs['gname'], return_counts=True)
for gname, c_id in group_dict:
    index = get_index(unique_groups,gname)
    if index == -1:
        print("WTF. " + gname)
    nevents = group_counts[index]
    #print(nevents)
    #nevents = nevents[0]
    for i in range(len(group_dict[(gname,c_id)])):
        group_dict[(gname,c_id)][i] = group_dict[(gname,c_id)][i]/nevents
    #group_dict[(gname,c_id)].append(nevents)

print(group_dict)
print(type(nevents))
print(nevents)

print(unique_groups)
print(group_counts)

dimension_arr = None # This should be a NumPy array 
for key_tup in group_dict:
    if dimension_arr == None:
        dimension_arr = np.asarray(group_dict[key_tup]) # First iteration, create the array
    else:
        dimension_arr = np.vstack((dimension_arr, group_dict[key_tup]))

print(dimension_arr)
from sklearn import preprocessing
dim_arr_scaled = preprocessing.scale(dimension_arr)
print(dim_arr_scaled)

from sklearn.cluster import KMeans
import numpy as np
k = 3
kmeans = KMeans(n_clusters=k,random_state=0).fit(dimension_arr)

print(len(group_dict))
print(len(group_dict.keys()))
print(len(kmeans.labels_))
#rint(group_dict.keys())
#rint(kmeans.labels_)
#i = 0
#for key_tup in group_dict:
#    group_dict[(key_tup[0],kmeans.labels_[i])] = group_dict.pop(key_tup) 
#    i += 1

#rint(group_dict)

print(len(group_dict))
print(len(kmeans.labels_))
print(dim_arr_scaled.shape)

result_dict = {}
for i in range(k):
    result_dict[i] = []

i = 0

for key_tup in group_dict:
    if i == 403:
        print("Wtf!" + str(group_dict[key_tup]))
        break
    #print(key_tup)
    #print(i)
    #print(group_dict[key_tup])
    #print(kmeans.labels_[i])
    result_dict[kmeans.labels_[i]].append(key_tup[0]) 
    i += 1
    

print(result_dict)

print(kmeans.cluster_centers_)

print(dimensions)

import operator
def print_largest_n(result, n):
    nevent_dict = {}
    for i in range(len(result)):
        index = get_index(unique_groups,result[i])
        nevents = group_counts[index]
        nevent_dict[result[i]] = nevents
    
    sorted_nevents_dict = sorted(nevent_dict.items(), key=operator.itemgetter(1),reverse=True)
    print(sorted_nevents_dict[:n])
    
for key in result_dict:
    print_largest_n(result_dict[key],5)



