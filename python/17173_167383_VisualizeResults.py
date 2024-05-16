import csv
import htmltag as HT
import DeriveFinalResultSet as drs
import JobsMapResultsFilesToContainerObjs as ImageMap
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import statistics as s
importlib.reload(drs)
importlib.reload(ImageMap)
pd.set_option('display.max_colwidth', -1)
from collections import Counter
import plotly.plotly as py
import cufflinks as cf
cf.go_offline()
import GetPropertiesAPI as GP
importlib.reload(drs)

def genHTMLTableFiles(shrCntsObj):
    shrPropDict = drs.getShrProp(shrCntsObj)
    
    df = pd.DataFrame(shrPropDict,index = ['Share Proportion']).transpose()
    
    return df,df.to_html(bold_rows = False)

# Generate rank list of all images by share proportion
rnkFlLst = []
with open("../FinalResults/rankListImages_expt2.csv","r") as rnkFl:
    rnkFlCsv = csv.reader(rnkFl)
    header = rnkFlCsv.__next__()
    for row in rnkFlCsv:
        rnkFlLst.append(row)
        
thTgs = []
trTgs = []

trTgs.append(HT.tr(HT.th("GID"),HT.th("Share count"),HT.th("Not Share count"),
                  HT.th("Total Count"),HT.th("Share Proportion"),HT.th("Image")))

for tup in rnkFlLst:
    tdGid = HT.td(tup[0])
    tdShare = HT.td(tup[1])
    tdNotShare = HT.td(tup[2])
    tdTot = HT.td(tup[3])
    tdProp = HT.td(tup[4])
    url = "https://socialmediabias.blob.core.windows.net/wildlifephotos/All_Zebra_Count_Images/" + tup[0] + ".jpeg"
    tdImg = HT.td(HT.img(src = url,alt = "Unavailable",width = "300"))
    trTgs.append(HT.tr(tdGid,tdShare,tdNotShare,tdTot,tdProp,tdImg))
    
fullFile = HT.html(HT.body(HT.table(HT.HTML('  \n'.join(trTgs)),border="1")))

outputFile = open("../data/resultsExpt2RankList1.html","w")
outputFile.write(fullFile)
outputFile.close()

# Generate the share prortion tables for pair wise features.

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","AGE",drs.imgJobMap,1,100)
h3_1 = HT.h3("Data-Frame by SPECIES-AGE")
df1,tb1 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","SEX",drs.imgJobMap,1,100)
h3_2 = HT.h3("Data-Frame by SPECIES-SEX")
df2,tb2 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","VIEW_POINT",drs.imgJobMap,1,100)
h3_3 = HT.h3("Data-Frame by SPECIES-VIEW_POINT")
df3,tb3 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","QUALITY",drs.imgJobMap,1,100)
h3_4 = HT.h3("Data-Frame by SPECIES-QUALITY")
df4,tb4 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SPECIES","EXEMPLAR_FLAG",drs.imgJobMap,1,100)
h3_5 = HT.h3("Data-Frame by SPECIES-EXEMPLAR_FLAG")
df5,tb5 = genHTMLTableFiles(d)

## *******## *******## *******## *******## *******## *******## *******## *******

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","SPECIES",drs.imgJobMap,1,100)
h3_6 = HT.h3("Data-Frame by VIEW_POINT-SPECIES")
df6,tb6 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","AGE",drs.imgJobMap,1,100)
h3_7 = HT.h3("Data-Frame by VIEW_POINT-AGE")
df7,tb7 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","SEX",drs.imgJobMap,1,100)
h3_8 = HT.h3("Data-Frame by VIEW_POINT-SEX")
df8,tb8 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","QUALITY",drs.imgJobMap,1,100)
h3_9 = HT.h3("Data-Frame by VIEW_POINT-QUALITY")
df9,tb9 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"VIEW_POINT","EXEMPLAR_FLAG",drs.imgJobMap,1,100)
h3_10 = HT.h3("Data-Frame by VIEW_POINT-EXEMPLAR_FLAG")
df10,tb10 = genHTMLTableFiles(d)

## *******## *******## *******## *******## *******## *******## *******## *******
d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","SPECIES",drs.imgJobMap,1,100)
h3_11 = HT.h3("Data-Frame by SEX-SPECIES")
df11,tb11 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","AGE",drs.imgJobMap,1,100)
h3_12 = HT.h3("Data-Frame by SEX-AGE")
df12,tb12 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","QUALITY",drs.imgJobMap,1,100)
h3_13 = HT.h3("Data-Frame by SEX-QUALITY")
df13,tb13 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","EXEMPLAR_FLAG",drs.imgJobMap,1,100)
h3_14 = HT.h3("Data-Frame by SEX-EXEMPLAR_FLAG")
df14,tb14 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"SEX","VIEW_POINT",drs.imgJobMap,1,100)
h3_15 = HT.h3("Data-Frame by SEX-VIEW_POINT")
df15,tb15 = genHTMLTableFiles(d)

## *******## *******## *******## *******## *******## *******## *******## *******
d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","SPECIES",drs.imgJobMap,1,100)
h3_16 = HT.h3("Data-Frame by AGE-SPECIES")
df16,tb16 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","SEX",drs.imgJobMap,1,100)
h3_17 = HT.h3("Data-Frame by AGE-SEX")
df17,tb17 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","QUALITY",drs.imgJobMap,1,100)
h3_18 = HT.h3("Data-Frame by AGE-QUALITY")
df18,tb18 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","EXEMPLAR_FLAG",drs.imgJobMap,1,100)
h3_19 = HT.h3("Data-Frame by AGE-EXEMPLAR_FLAG")
df19,tb19 = genHTMLTableFiles(d)

d = drs.ovrallShrCntsByTwoFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,"AGE","VIEW_POINT",drs.imgJobMap,1,100)
h3_20 = HT.h3("Data-Frame by AGE-VIEW_POINT")
df20,tb20 = genHTMLTableFiles(d)

fullFl = HT.html(HT.body(HT.HTML(h3_1),HT.HTML(tb1),
                        HT.HTML(h3_2),HT.HTML(tb2),
                        HT.HTML(h3_3),HT.HTML(tb3),
                        HT.HTML(h3_4),HT.HTML(tb4),
                        HT.HTML(h3_5),HT.HTML(tb5),
                        HT.HTML(h3_6),HT.HTML(tb6),
                        HT.HTML(h3_7),HT.HTML(tb7),
                        HT.HTML(h3_8),HT.HTML(tb8),
                        HT.HTML(h3_9),HT.HTML(tb9),
                        HT.HTML(h3_10),HT.HTML(tb10),
                        HT.HTML(h3_11),HT.HTML(tb11),
                        HT.HTML(h3_12),HT.HTML(tb12),
                        HT.HTML(h3_13),HT.HTML(tb13),
                        HT.HTML(h3_14),HT.HTML(tb14),
                        HT.HTML(h3_15),HT.HTML(tb15),
                        HT.HTML(h3_16),HT.HTML(tb16),
                        HT.HTML(h3_17),HT.HTML(tb17),
                        HT.HTML(h3_18),HT.HTML(tb18),
                        HT.HTML(h3_19),HT.HTML(tb19),
                        HT.HTML(h3_20),HT.HTML(tb20)
                        ))

outputFile = open("../FinalResults/twoFeatures.html","w")
outputFile.write(fullFl)
outputFile.close()

def getHTMLTabForFtr(ftr):
    d = drs.ovrallShrCntsByFtr(drs.gidAidMapFl,drs.aidFeatureMapFl,ftr,drs.imgJobMap,1,100)
    head3 = HT.h3("Data-Frame by " + ftr)
    df1,tb1 = genHTMLTableFiles(d)
    df1.sort_values(by=['Share Proportion'],ascending=False,inplace=True)
    fig = df1.iplot(kind='bar',filename=str(ftr + '_expt2' ))
    iframe = fig.embed_code

    df1.reset_index(inplace=True)
    df1.columns = [ftr,'Share Proportion']
    
    a,b,c = drs.genObjsForConsistency(drs.gidAidMapFl,drs.aidFeatureMapFl,ftr,drs.imgJobMap)    
    consistency = drs.getConsistencyDict(a,b,c)
    
    df2 = pd.DataFrame(drs.genVarStddevShrPropAcrsAlbms(consistency)).transpose()
    df2.reset_index(inplace=True)
    df2.columns = [ftr,'mean','standard_deviation','variance']
    
    df = pd.merge(df1,df2,left_on=ftr,right_on=ftr,how='left')
    
    return df,head3,df.to_html(bold_rows = False,index=False),iframe

# Generate the share prortion tables and visuals (bar diagrams) for single features.
df1,h3_1,tb1,img1 = getHTMLTabForFtr("SEX")
df2,h3_2,tb2,img2 = getHTMLTabForFtr("AGE")
df3,h3_3,tb3,img3 = getHTMLTabForFtr("SEX")
df4,h3_4,tb4,img4 = getHTMLTabForFtr("VIEW_POINT")
df5,h3_5,tb5,img5 = getHTMLTabForFtr("QUALITY")
df6,h3_6,tb6,img6 = getHTMLTabForFtr("EXEMPLAR_FLAG")
df7,h3_7,tb7,img7 = getHTMLTabForFtr("CONTRIBUTOR")
fullFl = HT.html(HT.body(HT.HTML(h3_1),HT.HTML(tb1), HT.html(img1),
                        HT.HTML(h3_2),HT.HTML(tb2), HT.html(img2),
                        HT.HTML(h3_3),HT.HTML(tb3), HT.html(img3),
                        HT.HTML(h3_4),HT.HTML(tb4), HT.html(img4),
                        HT.HTML(h3_5),HT.HTML(tb5), HT.html(img5),
                        HT.HTML(h3_6),HT.HTML(tb6), HT.html(img6),
                        HT.HTML(h3_7),HT.HTML(tb7), HT.html(img7)
                         ))

outputFile = open("../FinalResults/oneFeature.html","w")
outputFile.write(fullFl)
outputFile.close()

plt.close('all')

df = pd.DataFrame(drs.genAlbmFtrs(drs.gidAidMapFl,drs.aidFeatureMapFl,drs.imgJobMap,['SPECIES','AGE','SEX'])).transpose()
fullFl= HT.html(HT.body(HT.HTML(df.to_html(bold_rows = False))))

df.to_csv("../FinalResults/albumProperties.csv",index=False)
outputFile = open("../FinalResults/albumProperties.html","w")
outputFile.write(fullFl)
outputFile.close()

# DO NOT RUN AGAIN WITHOUT TAKING BACKUP OF THE HTML FILE
imgAlbmShrs,consistency = drs.getShrPropImgsAcrossAlbms(drs.imgJobMap,1,100,"../FinalResults/shareRateSameImgsAcrossAlbums.json")
df = pd.DataFrame(imgAlbmShrs,index=["Share Proportion"]).transpose()
gidShrVarStdDevDict = drs.genVarStddevShrPropAcrsAlbms(consistency)
df2 = pd.DataFrame(gidShrVarStdDevDict).transpose()
df2.reset_index(inplace=True)
df2.columns = ['GID','Standard Deviation','Variance']

subindex = df.groupby(level=0).head(1).index
subindex2 = df2.groupby(level=0).head(1)['Standard Deviation']
subindex3 = df2.groupby(level=0).head(1)['Variance']
df.loc[subindex, 'Standard Deviation'] = subindex2.get_values()
df.loc[subindex, 'Variance'] = subindex3.get_values()

df.to_csv("../FinalResults/shareRateSameImgsAcrossAlbums.csv")

df.loc[subindex, 'URL'] = '<img src = "https://socialmediabias.blob.core.windows.net/wildlifephotos/All_Zebra_Count_Images/' + subindex.get_level_values(0) + '.jpeg" width = "350">'
df = df.fillna("")

fullFl= HT.html(HT.body(HT.HTML(df.to_html(bold_rows = False))))
outputFile = open("../FinalResults/shareRateSameImgsAcrossAlbums.html","w")
outputFile.write(fullFl)
outputFile.close()

df = pd.DataFrame(sorted(gidShrVarStdDevDict.items(),key = lambda x : x[1]['standard_deviation'],reverse=True),columns=['GID','Stats'])
df.to_csv("../FinalResults/ImgsStdDevDesc.csv",index=False)

summaryPosCnt = drs.getPosShrProptn(drs.imgJobMap,1,100)

df = pd.DataFrame(summaryPosCnt).transpose()
cols = ['share','not_share','total','proportion']
df = df[cols]
# imgTg = HT.img(src="images/PositionBias.png")
# df.plot()
# plt.savefig("../FinalResults/PositionBias.png",bbox_inches='tight')

fig = df.iplot(kind='line',filename=str('Position_bias' + '_expt2' ))
iframe = fig.embed_code

fullFl= HT.html(HT.body(HT.HTML(df.to_html(bold_rows = False))),HT.HTML(iframe))
outputFile = open("../FinalResults/PositionBias1.html","w")
outputFile.write(fullFl)
outputFile.close()

df[['share','not_share','proportion']].plot()
plt.show()

# Overall share statistics ranked by shared proportion of images along with features and tags
df = ImageMap.createMstrFl("../data/resultsFeaturesComb_expt2.csv",[ 'GID', 'AID','Album', 'AGE','EXEMPLAR_FLAG', 'INDIVIDUAL_NAME', 'NID', 'QUALITY', 'SEX', 'SPECIES','VIEW_POINT','CONTRIBUTOR'])

dfRes = pd.DataFrame.from_csv("../FinalResults/resultsExpt2RankList_Tags.csv")
dfRes.reset_index(inplace=True)
dfRes.GID = dfRes.GID.astype(str)
dfRes['URL'] = '<img src = "https://socialmediabias.blob.core.windows.net/wildlifephotos/All_Zebra_Count_Images/' + dfRes['GID'] + '.jpeg" width = "350">'

dfSummary = pd.merge(df,dfRes,left_on ='GID',right_on='GID')
dfSummary.sort_values(by='Proportion',ascending=False,inplace=True)
dfSummary = dfSummary[['GID','Album','AGE','INDIVIDUAL_NAME','QUALITY','SEX','SPECIES','VIEW_POINT','CONTRIBUTOR','tags','Shared','Not Shared','Total','Proportion','URL']]
dfSummary.to_csv("/tmp/ImgShrRnkListWithTags.csv",index=False)

fullFl= HT.html(HT.body(HT.HTML(dfSummary.to_html(bold_rows = False,index=False))))
outputFile = open("../FinalResults/ImgShrRnkListWithTags.html","w")
outputFile.write(fullFl)
outputFile.close()

# Visualizations for general questions asked in the mechanical turk
ans = ImageMap.genCntrsGenQues(1,100,['Answer.q1','Answer.q2'])

q1 = ans['Answer.q1']
q1 = {key : q1[key] for key in q1 if key != ''}
dfQ1 = pd.DataFrame(q1,index=['Counts']).transpose()
dfQ1.sort_values(by='Counts',ascending=False,inplace=True)

fig = dfQ1.iplot(kind='bar',filename="Frequency of posting pictures",title="How frequently do you share pictures on social media")
iframe = fig.embed_code
iframe

mapVal = {'A' : 'None',
'B' : '1 to 5',
'C' : '5 to 10',
'D' : '10 to 50',
'E' : '50 or more'}

q2 = ans['Answer.q2']
q2 = {mapVal[key] : q2[key] for key in q2 if key != ''}
dfQ2 = pd.DataFrame(q2,index=['Counts']).transpose()
dfQ2.sort_values(by='Counts',ascending=False,inplace=True)

fig = dfQ2.iplot(kind='bar',filename="Number of photos people share after safari",title="How many photos will you share on social media after a safari")
iframe2 = fig.embed_code
iframe2



