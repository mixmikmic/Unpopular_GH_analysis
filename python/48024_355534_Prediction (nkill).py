# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat_for_nkill_pred.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])

print(dfs.columns)

dimensions = ['iyear', 'extended', 'success', 'suicide', 'gname', 'nperps', 'nkill', 'ishostkid', 'nhostkid',
              'weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 
               'weaptype1_txt_Fake Weapons', 'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee',
               'weaptype1_txt_Sabotage Equipment', 
               'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 
               'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 
               'attacktype1_txt_Facility/Infrastructure Attack', 'attacktype1_txt_Hijacking',
               'attacktype1_txt_Hostage Taking (Barricade Incident)', 'attacktype1_txt_Hostage Taking (Kidnapping)',
               'attacktype1_txt_Unarmed Assault', 'targtype1_txt_Abortion Related', 'targtype1_txt_Airports & Aircraft',
               'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 'targtype1_txt_Food or Water Supply', 
               'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)',
               'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime', 'targtype1_txt_Military',
               'targtype1_txt_NGO', 'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property',
               'targtype1_txt_Religious Figures/Institutions', 'targtype1_txt_Telecommunication', 
               'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 'targtype1_txt_Transportation',
               'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']

columns = dfs.columns
for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)

yarr = dfs['nkill']

del(dfs['nkill'])
del(dfs['gname'])
xarr = dfs.values.tolist()
print(type(xarr))   

xarr = np.array(xarr)
yarr = np.array(yarr)
print(type(xarr))
print(type(yarr))
print(xarr)
print(yarr)
xarr = np.nan_to_num(xarr)
yarr = np.nan_to_num(yarr)

from sklearn import model_selection

from sklearn import linear_model
from scipy.spatial import distance

def get_result_perc(result_dict, l):
    for key in result_dict:
        result_dict[key] = round(result_dict[key]*100/l,2)
    print(result_dict)
def prepare_result_dict():
    import collections
    result_dict = collections.OrderedDict()
    result_dict["1"] = 0
    result_dict["5"] = 0
    result_dict["10"] = 0
    result_dict["20"] = 0
    result_dict["50"] = 0
    result_dict["51+"] = 0
    return result_dict
def update_result_dict(result_dict, error_list):
    for error in error_list:
        if error < 1:
            result_dict["1"] += 1
        elif error < 5:
            result_dict["5"] += 1
        elif error < 10:
            result_dict["10"] += 1
        elif error < 20:
            result_dict["20"] += 1
        elif error < 50:
            result_dict["50"] += 1
        else:
            result_dict["51+"] += 1
    print(result_dict)

from sklearn.model_selection import KFold
k = 10
kf = KFold(n_splits=k,random_state=42)
kf.get_n_splits(xarr,yarr)
print(kf)  
eu_sum = 0
error_list = []
actual_list = []
rs_dict = prepare_result_dict()
actual_dict = prepare_result_dict()
for train_index, test_index in kf.split(xarr,yarr):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xarr[train_index], xarr[test_index]
    y_train, y_test = yarr[train_index], yarr[test_index]
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    yarr_model = regr.predict(X_test)
    eu_sum += distance.euclidean(y_test,yarr_model)
    for i in range(len(y_test)):
        error_list.append(abs(y_test[i]-yarr_model[i]))
        actual_list.append(y_test[i])

update_result_dict(rs_dict, error_list)
update_result_dict(actual_dict, actual_list)
get_result_perc(rs_dict,len(error_list))
get_result_perc(actual_dict,len(actual_list))
print(eu_sum)

import statistics
print(statistics.mean(error_list))
print(statistics.median(error_list))
print(statistics.mode(error_list))
print(statistics.variance(error_list))
print(max(error_list))
print(min(error_list))

get_ipython().magic('matplotlib inline')
df = pd.DataFrame.from_dict(rs_dict, orient='index')
g = df.plot(kind='bar', legend=False )
g.xaxis.set_label_text("Error less than")
g.yaxis.set_label_text("Percentage of events")
g.set_title("Frequency of errors")

import matplotlib.pyplot as plt
g = plt.scatter(error_list, actual_list)
g.axes.set_xlabel("Error Magnitude")
g.axes.set_ylabel("Actual value")
print("Error vs Actual Value")
plt.show()

err_list_copy = error_list.copy()
act_list_copy = actual_list.copy()

print(len(err_list_copy))
print(len(act_list_copy))
max_value = max(err_list_copy)
max_index = err_list_copy.index(max_value)
err_list_copy.remove(max_value)
act_list_copy.remove(act_list_copy[max_index])
max_value = max(err_list_copy)
max_index = err_list_copy.index(max_value)
err_list_copy.remove(max_value)
act_list_copy.remove(act_list_copy[max_index])

print(len(err_list_copy))
print(len(act_list_copy))

g = plt.scatter(err_list_copy, act_list_copy)
g.axes.set_xlabel("Error Magnitude")
g.axes.set_ylabel("Actual value")
print("Error vs Actual Value")
plt.show()



