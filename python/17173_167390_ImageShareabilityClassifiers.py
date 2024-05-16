import ClassiferHelperAPI as CH
import importlib
import numpy as np
import pandas as pd
importlib.reload(CH)
from ast import literal_eval
import plotly.plotly as py
import htmltag as HT
import cufflinks as cf # this is necessary to link pandas to plotly
cf.go_offline()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from collections import Counter
import csv
import plotly.graph_objs as go

allAttribs = CH.genAllAttribs("../FinalResults/ImgShrRnkListWithTags.csv","sparse","../data/infoGains.csv")
data= CH.getMasterData("../FinalResults/ImgShrRnkListWithTags.csv")

# Block of code for building and running the classifier
# Will generate custom warnings, setting scores to 0, if there are no valid predictions
methods = ['dummy', 'bayesian', 'logistic','svm','dtree','random_forests','ada_boost']
# methods = ['ada_boost']
clfArgs = {'dummy' : {'strategy' : 'most_frequent'},
            'bayesian' : {'fit_prior' : True},
            'logistic' : {'penalty' : 'l2'},
            'svm' : {'kernel' : 'rbf','probability' : True},
            'dtree' : {'criterion' : 'entropy'},
            'random_forests' : {'n_estimators' : 10 },
            'ada_boost' : {'n_estimators' : 50 }}
classifiers = []
for method in methods:
    for i in np.arange(0.4,0.5,0.1):
        clfObj = CH.buildBinClassifier(data,allAttribs,1-i,80,method,clfArgs)
        clfObj.runClf()
        classifiers.append(clfObj)

# Writing all the scores into a pandas data-frame and then into a CSV file
printableClfs = []

for clf in classifiers:
    printableClfs.append(dict(literal_eval(clf.__str__())))
    
df = pd.DataFrame(printableClfs)
df = df[['methodName','splitPercent','accScore','precision','recall','f1Score','auc','sqerr']]
df.columns = ['Classifier','Train-Test Split','Accuracy','Precision','Recall','F1 score','AUC','Squared Error']
# df.to_csv("../ClassifierResults/extrmClfMetrics_abv_mean.csv",index=False)

# Will take up valuable Plot.ly plots per day. Limited to 50 plots per day.
# changes to file name important
iFrameBlock = []
for i in np.arange(0.4,0.5,0.1):
    df1 = df[(df['Train-Test Split']==1-i)]
    df1.index = df1['Classifier']
    df1 = df1[['Accuracy','Precision','Recall','F1 score','AUC','Squared Error']].transpose()
    df1.iplot(kind='bar',filename=str('Train-Test_Split_Ratio_abv_mean %f' %i),title=str('Train-Test Split Ratio: %f' %i))
    # iFrameBlock.append(fig.embed_code)

# with open("../ClassifierResults/performanceComparisonsparse.html","w") as perf:
#     perf.write(HT.h1("Performance Comparisons of Classifiers with non_sparse Attributes."))
#     for row in iFrameBlock:
#         perf.write(HT.HTML(row))

clfWeights = []
for clf in classifiers:
    clfAttribs = list(clf.test_x.columns)
    if clf.methodName == 'logistic':
        clfAttribWgts = list(clf.clfObj.coef_[0])
    elif clf.methodName == 'dtree' or clf.methodName == 'random_forests':
        clfAttribWgts = list(clf.clfObj.feature_importances_)
    else:
        continue
        
        
    attribWgt = {clfAttribs[i] : clfAttribWgts[i] for i in range(len(clfAttribs))}
    attribWgt['Method'] = clf.methodName
    attribWgt['Split_Percent'] = clf.splitPercent
        
    clfWeights.append(attribWgt)

clfDf = pd.DataFrame(clfWeights)

indDF = clfDf[(clfDf['Method']=='logistic')]
indDF.index = indDF['Split_Percent']
indDF.drop('Method',1,inplace=True)  
indDF.transpose().to_csv("../ClassifierResults/LogisiticWeights.csv")

indDF = clfDf[(clfDf['Method']=='dtree')]
indDF.index = indDF['Split_Percent']
indDF.drop('Method',1,inplace=True)  
indDF.transpose().to_csv("../ClassifierResults/DecisionTreeWeights.csv")

indDF = clfDf[(clfDf['Method']=='random_forests')]
indDF.index = indDF['Split_Percent']
indDF.drop('Method',1,inplace=True)  
indDF.transpose().to_csv("../ClassifierResults/RandomForestsWeights.csv")

logisticDf = clfDf[(clfDf['Method']=='logistic')]
del logisticDf['Method']
del logisticDf['Split_Percent']
dtreeDf = clfDf[(clfDf['Method']=='dtree')]
del dtreeDf['Method']
del dtreeDf['Split_Percent']
randomForestDf = clfDf[(clfDf['Method']=='random_forests')]
del randomForestDf['Method']
del randomForestDf['Split_Percent']

logisticDf = logisticDf.transpose()
logisticDf.reset_index(inplace=True)
logisticDf.columns = ['Feature','10%','20%','30%','40%','50%','60%','70%','80%','90%']
dfs_logistic = []
for i in range(10,100,10):
    prcnt = str(i)+'%'
    logisticDf.sort_values(by=prcnt,inplace=True,ascending=False)
    df = logisticDf[['Feature',prcnt]].head(15)
    df.index = np.arange(1,16,1)
    
    dfs_logistic.append(df)
    
concatdf_logisitc = pd.concat([dfs_logistic[0],dfs_logistic[1],dfs_logistic[2],dfs_logistic[3],dfs_logistic[4],dfs_logistic[5],dfs_logistic[6],dfs_logistic[7],dfs_logistic[8]],axis=1)
concatdf_logisitc.to_csv("../ClassifierResults/Top15_Weights_Logisitic.csv")

dtreeDf = dtreeDf.transpose()
dtreeDf.reset_index(inplace=True)
dtreeDf.columns = ['Feature','10%','20%','30%','40%','50%','60%','70%','80%','90%']
dfs_tree = []
for i in range(10,100,10):
    prcnt = str(i)+'%'
    dtreeDf.sort_values(by=prcnt,inplace=True,ascending=False)
    df = dtreeDf[['Feature',prcnt]].head(15)
    df.index = np.arange(1,16,1)
    
    dfs_tree.append(df)
    
concatdf_dtree = pd.concat([dfs_tree[0],dfs_tree[1],dfs_tree[2],dfs_tree[3],dfs_tree[4],dfs_tree[5],dfs_tree[6],dfs_tree[7],dfs_tree[8]],axis=1)
concatdf_dtree.to_csv("../ClassifierResults/Top15_Weights_Dtree.csv")

randomForestDf = randomForestDf.transpose()
randomForestDf.reset_index(inplace=True)
randomForestDf.columns = ['Feature','10%','20%','30%','40%','50%','60%','70%','80%','90%']
dfs_rndf = []
for i in range(10,100,10):
    prcnt = str(i)+'%'
    randomForestDf.sort_values(by=prcnt,inplace=True,ascending=False)
    df = randomForestDf[['Feature',prcnt]].head(15)
    df.index = np.arange(1,16,1)
    
    dfs_rndf.append(df)
    
concatdf_rndf = pd.concat([dfs_rndf[0],dfs_rndf[1],dfs_rndf[2],dfs_rndf[3],dfs_rndf[4],dfs_rndf[5],dfs_rndf[6],dfs_rndf[7],dfs_rndf[8]],axis=1)
concatdf_rndf.to_csv("../ClassifierResults/Top15_Weights_Rndf.csv")

attribs = [list(dfs_logistic[i]['Feature']) for i in range(0,9)]
attribs = [attrib for listAttrib in attribs for attrib in listAttrib]
pd.DataFrame(Counter(attribs),index=['Frequency']).transpose().sort_values(by=['Frequency'],ascending=False)

attribs = [list(dfs_tree[i]['Feature']) for i in range(0,9)]
attribs = [attrib for listAttrib in attribs for attrib in listAttrib]
pd.DataFrame(Counter(attribs),index=['Frequency']).transpose().sort_values(by=['Frequency'],ascending=False)

attribs = [list(dfs_rndf[i]['Feature']) for i in range(0,9)]
attribs = [attrib for listAttrib in attribs for attrib in listAttrib]
pd.DataFrame(Counter(attribs),index=['Frequency']).transpose().sort_values(by=['Frequency'],ascending=False)

attribs = [list(dfs_logistic[i]['Feature']) for i in range(0,9)]
attribs += [list(dfs_tree[i]['Feature']) for i in range(0,9)]
attribs += [list(dfs_rndf[i]['Feature']) for i in range(0,9)]
attribs = [attrib for listAttrib in attribs for attrib in listAttrib]
pd.DataFrame(Counter(attribs),index=['Frequency']).transpose().sort_values(by=['Frequency'],ascending=False)

logisticDf.sort_values(by='10%',inplace=True,ascending=False)
fig = {
    'data' : [
        {'x' : logisticDf.Feature.head(15),'y' : logisticDf['10%'].head(15), 'mode' : 'markers', 'name' : '10%'}
    ]
}
iplot(fig)

obj1.precision

classifiers[0].preds

data= CH.getMasterData("../FinalResults/ImgShrRnkListWithTags.csv")  
methods = ['dummy','bayesian','logistic','svm','dtree','random_forests','ada_boost']
kwargsDict = {'dummy' : {'strategy' : 'most_frequent'},
            'bayesian' : {'fit_prior' : True},
            'logistic' : {'penalty' : 'l2'},
            'svm' : {'kernel' : 'rbf','probability' : True},
            'dtree' : {'criterion' : 'entropy'},
            'random_forests' : {'n_estimators' : 10 },
            'ada_boost' : {'n_estimators' : 50 }}

allAttribs = CH.genAllAttribs("../FinalResults/ImgShrRnkListWithTags.csv",'non_sparse',"../data/infoGainsExpt2.csv")
clfObj = CH.buildBinClassifier(data,allAttribs,1-0.5,80,'dtree',kwargsDict['dtree'])
clfObj.runClf()

clfObj.precision,clfObj.recall,clfObj.methodName

fpr,tpr,_ = clfObj.roccurve
rocCurve = {}
for i in range(len(fpr)):
    rocCurve[fpr[i]] = tpr[i]
    
pd.DataFrame(rocCurve,index=['tpr']).transpose().iplot()

CH.getLearningAlgo('random_forests', clfArgs)

classifiers[0].clfObj

df = df1.transpose().reset_index()
df

layout= go.Layout(
                    showlegend=True,
                    legend=dict(
                        x=1,
                        y=1,
                        font=dict(size=20)
                    ),
                    xaxis= dict(
                        title= 'Classification Quality Metrics',
                        ticklen= 5,
                        zeroline= True,
                        titlefont=dict(size=20),
                        tickfont=dict(size=20),
          # tickangle=45
                    ),
                    yaxis=dict(
                        ticklen= 5,
                        titlefont=dict(size=20),
                        tickfont=dict(size=20),
                        title="Percentage (%)"
                        #range=range
                    ),
        barmode='grouped'
                )

trace1 = go.Bar(
                    x = df1.index,
                    name = "Dummy",
                    y = df1['dummy']*100,
                    opacity = 0.5,
                    marker=dict(color='red')
                    
            )

trace2 = go.Bar(
                   x = df1.index,
                    name = "Bayesian",
                    y = df1['bayesian']*100,
                    opacity = 0.5,
                    marker=dict(color='green')
                    
            )

trace3 = go.Bar(
                   x = df1.index,
                    name = "Logistic",
                    y = df1['logistic']*100,
                    opacity = 0.5,
                    marker=dict(color='blue')
                    
            )

trace4 = go.Bar(
                   x = df1.index,
                    name = "SVM",
                    y = df1['svm']*100,
                    opacity = 1,
                    marker=dict(color='pink')
                    
            )

trace5 = go.Bar(
                   x = df1.index,
                    name = "Decision Tree",
                    y = df1['dtree']*100,
                    opacity = 1,
                    marker=dict(color='orange')
                    
            )

trace6 = go.Bar(
                   x = df1.index,
                    name = "Random Forests",
                    y = df1['random_forests']*100,
                    opacity = 0.5,
                    marker=dict(color='brown')
                    
            )

trace7 = go.Bar(
                   x = df1.index,
                    name = "Ada Boost",
                    y = df1['ada_boost']*100,
                    opacity = 1,
                    marker=dict(color='yellow')
                    
            )



data = [trace1, trace2, trace3,trace4,trace5,trace6, trace7]
fig = dict(data=data,layout=layout)
iplot(fig,filename="Expt2 Training data distributions")

df = df1.reset_index()

df1.index

df1

classifiers[0].roccurve[0]

layout= go.Layout(
                    showlegend=True,
                    legend=dict(
                        x=1,
                        y=1,
                        font=dict(size=15)
                    ),
                    xaxis= dict(
                        title= 'False Positive Rate (FPR)',
                        ticklen= 5,
                        zeroline= True,
                        titlefont=dict(size=15),
                        tickfont=dict(size=15),
          # tickangle=45
                    ),
                    yaxis=dict(
                        ticklen= 5,
                        titlefont=dict(size=15),
                        tickfont=dict(size=15),
                        title="True Positive Rate (TPR)"
                        #range=range
                    ),
        barmode='grouped'
                )

trace1 = go.Scatter(
                    x = classifiers[0].roccurve[0],
                    name = "Dummy",
                    y = classifiers[0].roccurve[1],
                    opacity = 0.5,
                    marker=dict(color='red')
                    
            )

trace2 = go.Scatter(
                    x = classifiers[1].roccurve[0],
                    name = "Bayesian",
                    y = classifiers[1].roccurve[1],
                    opacity = 0.5,
                    marker=dict(color='green')
                    
            )

trace3 = go.Scatter(
                    x = classifiers[2].roccurve[0],
                    name = "Logistic",
                    y = classifiers[2].roccurve[1],
                    opacity = 1,
                    marker=dict(color='blue')
                    
            )

trace4 = go.Scatter(
                    x = classifiers[3].roccurve[0],
                    name = "SVM",
                    y = classifiers[3].roccurve[1],
                    opacity = 0.5,
                    marker=dict(color='pink')
                    
            )

trace5 = go.Scatter(
                    x = classifiers[4].roccurve[0],
                    name = "Decision Tree",
                    y = classifiers[4].roccurve[1],
                    opacity = 1,
                    marker=dict(color='orange')
                    
            )

trace6 = go.Scatter(
                    x = classifiers[5].roccurve[0],
                    name = "Random Forests",
                    y = classifiers[5].roccurve[1],
                    opacity = 1,
                    marker=dict(color='brown')
                    
            )

trace7 = go.Scatter(
                    x = classifiers[6].roccurve[0],
                    name = "Ada Boost",
                    y = classifiers[6].roccurve[1],
                    opacity = 1,
                    marker=dict(color='yellow')
                    
            )



data = [trace1, trace2, trace3,trace4,trace5,trace6, trace7]
fig = dict(data=data,layout=layout)
iplot(fig,filename="Expt2 Training data distributions")



