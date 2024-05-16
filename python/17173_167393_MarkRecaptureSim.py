import json
from datetime import datetime
import DataStructsHelperAPI as DS
import JobsMapResultsFilesToContainerObjs as J
import importlib
importlib.reload(J)
import pandas as pd
import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_online()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import MarkRecapHelper as MR
import importlib
importlib.reload(MR)
import DeriveFinalResultSet as DRS
from collections import Counter

days = {'2015-02-18' : '2015-02-18',
 '2015-02-19' : '2015-02-19',
 '2015-02-20' : '2015-02-20',
 '2015-02-25' : '2015-02-25',
 '2015-02-26' : '2015-02-26',
 '2015-03-01' : '2015-03-01',
 '2015-03-02' : '2015-03-02'}

nidMarkRecapSet = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json",days)

# How many individuals were identified on each day, 
# i.e. how many different individuals did we see each day?

indsPerDay = {}
for nid in nidMarkRecapSet:
    for day in nidMarkRecapSet[nid]:
        indsPerDay[day] = indsPerDay.get(day,0) + 1
        
df1 = pd.DataFrame(indsPerDay,index=['IndsIdentified']).transpose()

fig1 = df1.iplot(kind='bar',filename='Individuals seen per day',title='Individuals seen per day')
iframe1 = fig1.embed_code

# How many individuals did we see only on that day, 
# i.e. how many individuals were only seen that day and not any other day.

uniqIndsPerDay = {}
for nid in nidMarkRecapSet:
    if len(nidMarkRecapSet[nid]) == 1:
        uniqIndsPerDay[nidMarkRecapSet[nid][0]] = uniqIndsPerDay.get(nidMarkRecapSet[nid][0],0) + 1
        
df2 = pd.DataFrame(uniqIndsPerDay,index=['IndsIdentifiedOnlyOnce']).transpose()

fig2 = df2.iplot(kind='bar',filename='Individuals seen only that day',title='Individuals seen only that day')
iframe2 = fig2.embed_code

# How many individuals were first seen on that day, i.e. the unique number of animals that were identified on that day.
# The total number of individuals across all the days is indeed equal to all the unique individuals in the database. We have 1997 identified individuals.
indsSeenFirst = {}
for nid in nidMarkRecapSet:
    indsSeenFirst[min(nidMarkRecapSet[nid])] = indsSeenFirst.get(min(nidMarkRecapSet[nid]),0) + 1
    
df3 = pd.DataFrame(indsSeenFirst,index=['FirstTimeInds']).transpose()

fig3 = df3.iplot(kind='bar',filename='Individuals first seen on that day',title='Individuals first seen on that day')
iframe3 = fig3.embed_code

df1['IndsIdentifiedOnlyOnce'] = df2['IndsIdentifiedOnlyOnce']
df1['FirstTimeInds'] = df3['FirstTimeInds']

df1.columns = ['Total inds seen today','Inds seen only today','Inds first seen today']
fig4 = df1.iplot(kind='bar',filename='Distribution of sightings',title='Distribution of sightings')
iframe4 = fig4.embed_code

days = {'2015-03-01' : 1,
        '2015-03-02' : 2 }

# Entire population estimate (includes giraffes and zebras)
nidMarkRecapSet = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json","../FinalResults/rankListImages_expt2.csv",days,shareData=None)
marks,recaptures,population = MR.applyMarkRecap(nidMarkRecapSet)
print("Population of all animals = %f" %population)
marks,recaptures

nidMarkRecapSet_Zebras = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json","../FinalResults/rankListImages_expt2.csv",days,'zebra_plains',shareData=None)
marks,recaptures,population = MR.applyMarkRecap(nidMarkRecapSet_Zebras)
print("Population of zebras = %f" %population)
marks,recaptures

nidMarkRecapSet_Giraffes = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json","../FinalResults/rankListImages_expt2.csv",days,'giraffe_masai',shareData=None)
marks,recaptures,population = MR.applyMarkRecap(nidMarkRecapSet_Giraffes)
print("Population of giraffes = %f" %population)
marks,recaptures

nidMarkRecapSet_share = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json",
                       "../data/full_gid_aid_map.json",
                       "../data/full_aid_features.json",
                       "../FinalResults/rankListImages_expt2.csv",
                       days,
                       None,
                       shareData='proportion')
mark,recapture,population = MR.applyMarkRecap(nidMarkRecapSet_share)
print("Population of all animals = %f" %population)
marks,recaptures

nidMarkRecapSet_share = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json",
                       "../data/full_gid_aid_map.json",
                       "../data/full_aid_features.json",
                       "../FinalResults/rankListImages_expt2.csv",
                       days,
                       'zebra_plains',
                       shareData='proportion')
mark,recapture,population = MR.applyMarkRecap(nidMarkRecapSet_share)
print("Population of zebras = %f" %population)
marks,recaptures

nidMarkRecapSet_share = MR.genNidMarkRecapDict("../data/imgs_exif_data_full.json",
                       "../data/full_gid_aid_map.json",
                       "../data/full_aid_features.json",
                       "../FinalResults/rankListImages_expt2.csv",
                       days,
                       'giraffe_masai',
                       shareData='proportion')
mark,recapture,population = MR.applyMarkRecap(nidMarkRecapSet_share)
print("Population of giraffes = %f" %population)
marks,recaptures

days = [{'2004' : 1, '2005' : 2 },{'2005' : 1, '2006' : 2 }, {'2006' : 1, '2007' : 2 }, {'2007' : 1, '2008' : 2 }, {'2008' : 1, '2009' : 2 }, {'2009' : 1, '2010' : 2 }, {'2010' : 1, '2011' : 2 }, {'2014' : 1, '2015' : 2 }, {'2015' : 1, '2016' : 2}, {'2016' : 1, '2017' : 2}] 
for i in range(len(days)):
    nidMarkRecapSet = MR.genNidMarkRecapDict("/tmp/gir_new_exif.json",
                                         "../data/Flickr_IBEIS_Ftrs_gid_aid_map.json",
                                         "../data/Flickr_IBEIS_Giraffe_Ftrs_aid_features.json",
                                         "../FinalResults/rankListImages_expt2.csv", # this is useless
                                         days[i],
                                         shareData='other',
                                        filterBySpecies='giraffe_reticulated')
    marks, recaps, population, confidence = MR.applyMarkRecap(nidMarkRecapSet)
      
    print("Estimate for the year : "  + ' & '.join(list(days[i].keys())))
    print("Number of marks : %i" %marks)
    print("Number of recaptures : %i" %recaps)
    print("Estimated population : %f" %population)
    print()

inGidAidMapFl, inAidFtrFl = "../data/Flickr_IBEIS_Ftrs_gid_aid_map.json", "../data/Flickr_IBEIS_Ftrs_aid_features.json",

gidNid = DRS.getCountingLogic(inGidAidMapFl,inAidFtrFl,"NID",False)
flickr_nids = list(gidNid.values())
flickr_nids = [item for sublist in flickr_nids for item in sublist]

print("Number of unique individuals identified : %i" %len(set(flickr_nids)))

occurence = Counter(flickr_nids)

inExifFl = "../data/Flickr_EXIF_full.json"
with open(inExifFl, "r") as fl:
    obj = json.load(fl)

'''
lat in between -1.50278 and 1.504953
long in between 35.174045 and 38.192836
'''

gids_geoTagged = [gid for gid in obj.keys() if int(gid) < 1702 and obj[gid]['lat'] != 0 ]
gids_nairobi = [gid for gid in obj.keys() if int(gid) <1702 and obj[gid]['lat'] >= -1.50278 and obj[gid]['lat'] <= 1.504953 and obj[gid]['long'] >= 35.174045 and obj[gid]['long'] <= 38.192836 ]
gids_zoo = list(set(gids_geoTagged) - set(gids_nairobi))

import DeriveFinalResultSet as DRS, DataStructsHelperAPI as DS

inGidAidMapFl, inAidFtrFl = "../data/Flickr_IBEIS_Ftrs_gid_aid_map.json", "../data/Flickr_IBEIS_Ftrs_aid_features.json",

gidNid = DRS.getCountingLogic(inGidAidMapFl,inAidFtrFl,"NID",False)

locs = []
for gid in gidNid.keys():
    if gid in gids:
        for nid in gidNid[gid]:
            locs.append((obj[gid]['lat'], obj[gid]['long']))

nid_gid = DS.flipKeyValue(gidNid)

nids_zoo = []

for gid in gidNid.keys():
    if gid in gids_zoo:
        nids_zoo.extend(gidNid[gid])

len(gids_zoo), len(nids_zoo)

# removing all nids that are in zoos, with it you will also remove the other occurences of images in which that individual occurs.
nids_only_wild_gid =  {nid : nid_gid[nid] for nid in nid_gid.keys() if nid not in nids_zoo}
nids_zoo_wild_gid = {nid : nid_gid[nid] for nid in nid_gid.keys() if nid in nids_zoo}

len(list(nids_only_wild_gid.values())), len(nids_zoo_wild_gid.values())

len({gid for sublist in list(nids_only_wild_gid.values()) for gid in sublist})

len({gid for sublist in list(nids_zoo_wild_gid.values()) for gid in sublist})

max(list(map(int, list(gidNid.keys()))))

gidNid['110']

l =[12,12,12,12,12]
l.extend([1,2,3])

a = 5

print("a = %d" %a)

MR.genNidMarkRecapDict("../data/Flickr_Giraffe_EXIF.json",
                                         "../data/Flickr_IBEIS_Ftrs_gid_aid_map.json",
                                         "../data/Flickr_IBEIS_Giraffe_Ftrs_aid_features.json",
                                         "../FinalResults/rankListImages_expt2.csv", # this is useless
                                         days[i],
                                         shareData='other',
                                        filterBySpecies='giraffe_reticulated')

gidSpecies

{ gid : gidsDayNumFull[gid] for gid in gidsDayNumFull  if gid in gidSpecies.keys() and 'giraffe_reticulated' in gidSpecies[gid]}

gidsDayNumFull

gir_exif= DS.json_loader("../data/Flickr_Giraffe_EXIF.json")
gid_fl = DS.json_loader("../data/Flickr_Giraffes_imgs_gid_flnm_map.json")

fl_gid = DS.flipKeyValue(gid_fl)

new_exif = {}
for fl in gir_exif.keys():
    new_exif[fl_gid[fl]] = gir_exif[fl]

with open("/tmp/gir_new_exif.json", "w") as fl:
    json.dump(new_exif, fl, indent=4)

df = pd.DataFrame(new_exif).transpose()

df['date'] = pd.to_datetime(df['date'])

df['year'] = df.date.apply(lambda x : x.year)

year_num = dict(df.groupby('year').count()['date'])

year_num = [(key,year_num[key]) for key in year_num.keys()]

X = [year_num[i][0] for i in range(len(year_num)) if year_num[i][0] > 1999]
Y = [year_num[i][1] for i in range(len(year_num)) if year_num[i][0] > 1999]

import plotly.graph_objs as go

data = [go.Bar(
            x=X,
            y=Y
    )]
layout = go.Layout(
    annotations=[
        dict(x=xi,y=yi,
             text=str(yi),
             xanchor='center',
             yanchor='bottom',
             showarrow=False,
        ) for xi, yi in zip(X, Y)]
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='basic-bar')

year_num[0][1]



