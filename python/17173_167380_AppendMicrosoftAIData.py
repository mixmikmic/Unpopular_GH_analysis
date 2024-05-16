import csv
import json
import JobsMapResultsFilesToContainerObjs as ImageMap
import DeriveFinalResultSet as drs
import DataStructsHelperAPI as DS
import importlib
import pandas as pd
import htmltag as HT
from collections import OrderedDict
#import matplotlib.pyplot as plt
import plotly.plotly as py
import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_online()
flName = "../data/All_Zebra_Count_Tag_Output_Results.txt"
pd.set_option('display.max_colwidth', -1)
imgAlbumDict = ImageMap.genImgAlbumDictFromMap(drs.imgJobMap)
master = ImageMap.createResultDict(1,100)
imgShareNotShareList,noResponse = ImageMap.imgShareCountsPerAlbum(imgAlbumDict,master)
importlib.reload(ImageMap)
importlib.reload(DS)

header,rnkFlLst = DS.genlstTupFrmCsv("../FinalResults/rankListImages_expt2.csv")
rnkListDf = pd.DataFrame(rnkFlLst,columns=header)
rnkListDf['Proportion'] = rnkListDf['Proportion'].astype('float')
rnkListDf.sort_values(by="Proportion",ascending=False,inplace=True)

# create an overall giant csv
gidFtrs = ImageMap.genMSAIDataHighConfidenceTags("../data/GZC_data_tagged.json",0.5)
        
gidFtrsLst = DS.cnvrtDictToLstTup(gidFtrs)
df = pd.DataFrame(gidFtrsLst,columns=['GID','tags'])

shrPropsTags = pd.merge(rnkListDf,df,left_on='GID',right_on='GID')

# shrPropsTags.to_csv("../FinalResults/resultsExpt2RankList_Tags.csv",index=False)
shrPropsTags['URL'] = '<img src = "https://socialmediabias.blob.core.windows.net/wildlifephotos/All_Zebra_Count_Images/' + shrPropsTags['GID'] + '.jpeg" width = "350">'

shrPropsTags.sort_values(by=['Proportion','GID'],ascending=False,inplace=True)
fullFl = HT.html(HT.body(HT.HTML(shrPropsTags.to_html(bold_rows = False,index=False))))

fullFl
# outputFile = open("../FinalResults/resultsExpt2RankList_Tags.html","w")
# outputFile.write(fullFl)
# outputFile.close()

tgsShrNoShrCount = {}
for lst in rnkFlLst:
    tgs = gidFtrs[lst[0]]
    tmpDict = {'share': int(lst[1]), 'not_share': int(lst[2]), 'total' : int(lst[3])}
    for tag in tgs:
        oldDict ={}
        oldDict =  tgsShrNoShrCount.get(tag,{'share' : 0,'not_share' : 0,'total' : 0})
        oldDict['share'] = oldDict.get('share',0) + tmpDict['share']
        oldDict['not_share'] = oldDict.get('not_share',0) + tmpDict['not_share']
        oldDict['total'] = oldDict.get('total',0) + tmpDict['total']

        tgsShrNoShrCount[tag] = oldDict

## Append data into data frames and build visualizations
tgsShrCntDf = pd.DataFrame(tgsShrNoShrCount).transpose()
tgsShrCntDf['proportion'] = tgsShrCntDf['share'] * 100 / tgsShrCntDf['total']
tgsShrCntDf.sort_values(by=['proportion','share'],ascending=False,inplace=True)
tgsShrCntDf = tgsShrCntDf[['share','not_share','total','proportion']]
tgsShrCntDf.to_csv("../FinalResults/RankListTags.csv")

fullFl = HT.html(HT.body(HT.HTML(tgsShrCntDf.to_html(bold_rows = False))))

outputFile = open("../FinalResults/RankListTags.html","w")
outputFile.write(fullFl)
outputFile.close()

iFrameBlock = []
fig = tgsShrCntDf['proportion'].iplot(kind='line',filename="All_Tags",title="Distribution of Tags")
iFrameBlock.append(fig.embed_code)
#plt.savefig("../FinalResults/RankListTags.png",bbox_inches='tight')

gidFtrs = ImageMap.genMSAIDataHighConfidenceTags("../data/GZC_data_tagged.json",0.5)
        
gidFtrsLst = DS.cnvrtDictToLstTup(gidFtrs)
df = pd.DataFrame(gidFtrsLst,columns=['GID','tags'])

df



