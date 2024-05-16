import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_offline()
import json
import plotly.graph_objs as go
import pandas as pd
import htmltag as HT
import PopulationEstimatorAPI as PE, ClassiferHelperAPI as CH
import importlib
import MarkRecapHelper as MR
importlib.reload(PE)
import random
import DataStructsHelperAPI as DS
from plotly.offline import plot, iplot

attribs = [ 'GID', 'AID', 'AGE',
       'EXEMPLAR_FLAG', 'INDIVIDUAL_NAME', 'NID', 'QUALITY', 'SEX', 'SPECIES',
       'VIEW_POINT','CONTRIBUTOR']

df = ImageMap.genGidAidFtrDf("../data/full_gid_aid_map.json","../data/full_aid_features.json",'../data/full_gid_aid_ftr.csv')
df_comb = ImageMap.createMstrFl("../data/full_gid_aid_ftr.csv","../data/GZC_data_tagged.json",attribs,"../data/full_gid_aid_ftr_agg.csv")

with open("../FinalResults/PopulationEstimate.json","r") as jsonFl:
    resObj = json.load(jsonFl)

df = pd.DataFrame(resObj)
df['Axes Name'] = df['Classifier'] + " " + df['Attribute']

df = df[['Axes Name', 'all','giraffes','zebras','shared_images_count']]
df['Error_total_pop'] = df['all'] - 3620
df['Error_zebra_pop'] = df['zebras'] - 3468
df['Error_giraffe_pop'] = df['giraffes'] - 177
df['Predicted_Shared_proportion'] = df['shared_images_count'] * 100 / 6523
dfFull = df[['Axes Name','all','Error_total_pop','zebras','Error_zebra_pop','giraffes','Error_giraffe_pop','shared_images_count','Predicted_Shared_proportion']]
dfFull['norm_error_total_pop'] = dfFull['Error_total_pop'] / 3620
dfFull['norm_error_zebra_pop'] = dfFull['Error_zebra_pop'] / 3468
dfFull['norm_error_giraffe_pop'] = dfFull['Error_giraffe_pop'] / 177
dfFull.head()

dfErrors= dfFull[['Axes Name','Error_total_pop','Error_zebra_pop','Error_giraffe_pop']]
dfErrors.index = df['Axes Name']
dfErrors.drop(['Axes Name'],1,inplace=True)

layout = go.Layout(
    title="Estimation absolute-errors using predict-shared data",
    titlefont = dict(
            size=22),
    xaxis=dict(
        title="Classifier and Attribute Selection method",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickangle=35,
        tickfont=dict(
            size=9,
            color='black')
    ),
    yaxis=dict(
        title="Absolute Error",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickfont=dict(
            size=9,
            color='black')
    ))
fig1 = dfErrors.iplot(kind='bar',filename="Absolute_Errors",layout=layout)

dfNormErrors= dfFull[['Axes Name','norm_error_total_pop','norm_error_zebra_pop','norm_error_giraffe_pop']]
dfNormErrors.index = df['Axes Name']
dfNormErrors.drop(['Axes Name'],1,inplace=True)

layout = go.Layout(
    title="Estimation normalized-errors using predict-shared data",
    titlefont = dict(
            size=22),
    xaxis=dict(
        title="Classifier and Attribute Selection method",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickangle=35,
        tickfont=dict(
            size=9,
            color='black')
    ),
    yaxis=dict(
        title="Normalized Error",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickfont=dict(
            size=9,
            color='black')
    ))
fig2 = dfNormErrors.iplot(kind='bar',filename="Norm_Errors",layout=layout)
# Error = (predicted population - actual population)
# Normalized error formula =  Error / actual population

dfNoOutliers = dfErrors[(abs(dfErrors['Error_total_pop']) <= 2750 )][(abs(dfErrors['Error_total_pop']) > 10)]

layout = go.Layout(
    title="Estimation errors using predict-shared data -no outliers",
    titlefont = dict(
            size=22),
    xaxis=dict(
        title="Classifier and Attribute Selection method",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickangle=35,
        tickfont=dict(
            size=9,
            color='black')
    ),
    yaxis=dict(
        title="Absolute Error",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickfont=dict(
            size=9,
            color='black')
    ))
fig3 = dfNoOutliers.iplot(kind='bar',filename="errors_noOutliers",layout=layout)

# predicted shared proportion (x) vs normalized error zebra (y1) and giraffe (y2)? thanks!
dfNewPlot = dfFull[['Predicted_Shared_proportion','norm_error_zebra_pop','norm_error_giraffe_pop']]
dfNewPlot.index = dfNewPlot['Predicted_Shared_proportion']/100
dfNewPlot.drop(['Predicted_Shared_proportion'],1,inplace=True)
dfNewPlot.head()

layout = go.Layout(
    title="Predicted Shared Proportion versus Norm Error",
    titlefont = dict(
            size=22),
    xaxis=dict(
        title="Predicted Share Proportion",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickangle=35,
        tickfont=dict(
            size=9,
            color='black')
    ),
    yaxis=dict(
        title="Normalized Error",
        titlefont = dict(
            size=15),
        showticklabels=True,
        tickfont=dict(
            size=9,
            color='black')
    )
    )
fig4 = dfNewPlot.iplot(kind='bar',filename="predictedSharedVsError",layout=layout)

fullFl = HT.HTML(HT.body(HT.h2("Population Estimates using predicted shared data - master table"),
                HT.HTML(dfFull.to_html(index=False)),
                HT.HTML(fig1.embed_code),
                HT.HTML(fig2.embed_code),
                HT.HTML(fig3.embed_code),
                HT.HTML(fig4.embed_code)
               ))


outputFile = open("../FinalResults/PopulationEstimationUsingClf.html","w")
outputFile.write(fullFl)
outputFile.close()

appearanceDays = {}
for card in sdCards.keys():
    pred_results = {gid : predResults[gid] for gid in sdCards[card] if gid != '3644'}
    dfPredRes = pd.DataFrame(pred_results,index=['share']).transpose().reset_index()
    dfPredRes.columns = ['GID','share']
    appearanceDays[card] = set(pd.DataFrame.merge(dfPredRes,dfGidDays,on='GID').to_dict()['day'].values())

appearanceDays

import PopulationEstimatorAPI as PE
import importlib
importlib.reload(PE)

l = PE.buildErrPlots('clf')
for ifrm in l:
    print(ifrm)
    print("<p>X-axis : k <br>Y axis = Percentage Error</p>")
    print()

import pandas as pd

def buildErrPlots(clfOrRgr, thresholdMeth=False, randomShare=False):
    if clfOrRgr == 'clf':
        algTypes = ['bayesian','logistic','svm','dtree','random_forests','ada_boost']
    else:
        algTypes = ['linear','ridge','lasso','svr','dtree_regressor','elastic_net']
    attribTypes = ['sparse','non_sparse','non_zero','abv_mean', 'beauty']
    
    flNms = [str(alg + "_" + attrib) for alg in algTypes for attrib in attribTypes]

    if thresholdMeth:
        suffix = "_thresholded.csv"
        hdr = "threshold"
        if clfOrRgr == 'clf':
            titleSuffix = "classifiers thresholded"
        else:
            titleSuffix = "regressors thresholded"
    else:
        hdr = "num_images"
        if randomShare:
            suffix = "_kSharesRandom.csv"
            if clfOrRgr == 'clf':
                titleSuffix = "classifiers random choices"
            else:
                titleSuffix = "regressors random choices"
        else:
            suffix = "_kShares.csv"
            if clfOrRgr == 'clf':
                titleSuffix = "classifiers top k choices"
            else:
                titleSuffix = "regressors top k choices"

    df = pd.DataFrame.from_csv(str("../FinalResults/"+flNms[0]+suffix)).reset_index()
    df.columns = list(map(lambda x : str(x + "_" + flNms[0]) if x != hdr else x,list(df.columns)))
    for i in range(1,len(flNms)):
        df1 = pd.DataFrame.from_csv(str("../FinalResults/"+flNms[i]+suffix)).reset_index()
        df1.columns = list(map(lambda x : str(x + "_" + flNms[i]) if x != hdr else x,list(df1.columns)))
        df = pd.DataFrame.merge(df,df1,on=hdr)

    df.index = df[hdr]
    df.drop([hdr],1,inplace=True)
    

    # calculate errors in estimation
    # % error = (predicted - actual) * 100 / actual
    for col in df.columns:
        if 'all' in col:
            df[str(col+'_err')] = (df[col] - 3620) / 36.20
        elif 'zebras' in col:
            df[str(col+'_err')] = (df[col] - 3468) / 34.68
        elif 'giraffes' in col:
            df[str(col+'_err')] = (df[col] - 177) / 1.77

    figs=[]
    errorCols = [col for col in df.columns if 'err' in col]
    # df = df[errorCols]
    return df

    for alg in algTypes:
        algCol = [col for col in df.columns if alg in col]
        algDf = df[algCol]
        titleAlg = "All %s %s" %(alg,titleSuffix)
        figs.append(algDf.iplot(kind='line',title=titleAlg))

    for attrib in attribTypes:
        attribCol = [col for col in df.columns if attrib in col]
        attribDf = df[attribCol]
        titleAttrib = "All %s %s" %(attrib,titleSuffix)
        figs.append(attribDf.iplot(kind='line',title=titleAttrib))

    figCodes = [fig.embed_code for fig in figs]
    return figCodes

df = buildErrPlots('clf', randomShare=True)

df.to_csv("/tmp/test.csv")



cols = list(filter(lambda x : 'zebra' in x and 'beauty' in x, list(df.columns)))
df[cols].to_csv("/tmp/zebras_bty_rgr.csv")

import PopulationEstimatorAPI as PE
import importlib
importlib.reload(PE)

l = PE.buildErrPlots('rgr', thresholdMeth=True)

for i in l:
    print(i)
    print("<p>X-axis : k <br>Y axis = Percentage Error</p>")
    print()

train_fl, test_fl = "../data/BeautyFtrVector_GZC_Expt2.csv", "../data/GZC_exifs_beauty_full.csv"
inExifFl,inGidAidMapFl,inAidFtrFl = "../data/imgs_exif_data_full.json","../data/full_gid_aid_map.json","../data/full_aid_features.json"
meth = 'linear'
attrib = 'beauty'
regrArgs = {'linear' : {'fit_intercept' : True},
            'ridge' : {'fit_intercept' : True},
            'lasso' : {'fit_intercept' : True},
            'elastic_net' : {'fit_intercept' : True},
            'svr' : {'fit_intercept' : True},
            'dtree_regressor' : {'fit_intercept' : True}}

methObj,predResults = CH.trainTestRgrs(train_fl,
                                test_fl,
                                meth,
                                attrib,
                                infoGainFl="../data/infoGainsExpt2.csv",
                                methArgs = regrArgs
                                )

PE.kSharesPerContribAfterCoinFlip(predResults, inExifFl, inGidAidMapFl, inAidFtrFl, lambda : 2)

res = [{'all': 1320.0, 'giraffes': None, 'zebras': 817.0},
{'all': 2000.0, 'giraffes': 120, 'zebras': 817.0},
{'all': 2220.0, 'giraffes': None, 'zebras': None},
{'all': 3220.0, 'giraffes': 180, 'zebras': 2000},
{'all': 3220.0, 'giraffes': 180, 'zebras': 2500}]

df1 = pd.DataFrame(res)

df1.iplot(kind='line')

df = PE.runSyntheticExptsRgr(inExifFl, inGidAidMapFl, inAidFtrFl, range(2,30), thresholdMeth=False, randomShare=False, beautyFtrs = True)

df['gnd_truth_zebra'] = 3468
df['gnd_truth_girrafe'] = 177
df['gnd_truth_all'] = 3628

df.plot(kind='line')

import matplotlib.pyplot as plt

plt.show()

df.drop(['all', 'giraffes', 'gnd_truth_girrafe', 'gnd_truth_all'],1,inplace=True)

df.head()



