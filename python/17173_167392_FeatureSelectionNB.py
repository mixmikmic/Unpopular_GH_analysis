import ClassiferHelperAPI as CH
import pandas as pd
import sys
import FeatureSelectionAPI as FS
import importlib
importlib.reload(FS)
from numpy import mean
import csv

# Generating attributes, converting categorical attributes into discrete binary output.
# For instance - SPECIES : Zebra will be converted into (Zebra: 1, Giraffe: 0 .. )
hasSparse = False
data= CH.getMasterData("../FinalResults/ImgShrRnkListWithTags.csv")
if hasSparse:   
    ftrList = ['SPECIES','SEX','AGE','QUALITY','VIEW_POINT','INDIVIDUAL_NAME','CONTRIBUTOR','tags'] 
else:
    ftrList = ['SPECIES','SEX','AGE','QUALITY','VIEW_POINT'] #,'tags']
    
allAttribs = CH.genAttribsHead(data,ftrList)

ftrList = ['INDIVIDUAL_NAME','CONTRIBUTOR','tags'] 
allAttribs = CH.genAttribsHead(data,ftrList)

gidAttribDict = CH.createDataFlDict(data,allAttribs,80,'Train') # binaryClf attribute in createDataFlDict will be True here

df = pd.DataFrame.from_dict(gidAttribDict).transpose()
df = df[allAttribs+["TARGET"]]
df.head(5)

infoGains = [(col,FS.infoGain(df[col],df.TARGET)) for col in df.columns]

for col in df.columns:
    infoGains.append((col,FS.infoGain(df[col],df.TARGET)))
infoGains = sorted(infoGains,key = lambda x : x[1],reverse=True)
infoGains = infoGains[2:]
infoGains

with open("../data/infoGainsExpt2.csv","w") as infGainFl:
    csvWrite = csv.writer(infGainFl)
    
    for row in infoGains:
        csvWrite.writerow(row)

len(infoGains)

import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_offline()
import pandas as pd

d = [(1,2),(2,3),(3,4)]

pd.DataFrame(d).iplot()



