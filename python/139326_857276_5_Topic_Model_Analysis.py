import urllib.request
import os

def download_file_from_blob(filename):
    shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
    save_path = os.path.join(shared_path, filename)

    if not os.path.exists(save_path):
        # Base URL for anonymous read access to Blob Storage container
        STORAGE_CONTAINER = 'https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/'
        url = STORAGE_CONTAINER + filename
        urllib.request.urlretrieve(url, save_path)
        print("Downloaded file: %s" % filename)
    else:
        print("File \"%s\" already existed" % filename)

download_file_from_blob('CongressionalDocsLDA.pickle')
download_file_from_blob('CongressionalDocsLDA.pickle.expElogbeta.npy')
download_file_from_blob('CongressionalDocsLDA.pickle.id2word')
download_file_from_blob('CongressionalDocsLDA.pickle.state')
download_file_from_blob('CongressionalDocsLDA.pickle.state.sstats.npy')
download_file_from_blob('CongressionalDocTopicLM.npy')
download_file_from_blob('CongressionalDocTopicProbs.npy')
download_file_from_blob('CongressionalDocTopicSummaries.tsv')
download_file_from_blob('Vocab2SurfaceFormMapping.tsv')

import numpy as np
import pandas 
import re
import math
import warnings
import os
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
from operator import itemgetter
from collections import namedtuple
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from azureml.logging import get_azureml_logger
aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
aml_logger.log('amlrealworld.document-collection-analysis.notebook5', 'true')

# Load full TSV file including a column of text
docsFrame = pandas.read_csv(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDataAll_Jun_2017.tsv"), 
                            sep='\t')

print(docsFrame[90:100])

# Break out the session number as a unique column in the documents frame 
# also create a column for the session quarter where we break dates down 
# into one of eight annual quarters per session, i.e., four quarters for 
# each of the two years in the session

reType = re.compile(r"^([a-z]+)[0-9]+$")

quarterArray = [] 
sessionArray = []
typeArray = []
for i in range(0,len(docsFrame)):

    dateFields = (docsFrame['Date'][i]).split('-')
    year = int(dateFields[0])
    month = int(dateFields[1])
    evenYear = int ((year % 2) == 0) 
    quarterArray.append(int((month - 1) / 3 ) + (evenYear * 4))

    idFields = (docsFrame['ID'][i]).split('-')
    
    billType = reType.match(idFields[0]).group(1)
    typeArray.append(billType)
    session = int(idFields[1])
    sessionArray.append(session)

# Add the meta-data entries into the data frame
docsFrame['Quarter'] = quarterArray
docsFrame['Session'] = sessionArray
docsFrame['Type'] = typeArray

# Extract the minimum session number in the data
minSessionNum = min(sessionArray)  

sessionQuarterIndex = []
for i in range(len(docsFrame)):
    session = docsFrame['Session'][i]
    quarter = docsFrame['Quarter'][i]
    sessionQuarterIndex.append(((session-minSessionNum)*8)+quarter)
    
docsFrame['SessionQuarterIndex'] = sessionQuarterIndex
maxSessionQuarterIndex = max(sessionQuarterIndex)
print("Total number of quarters over all sessions in data:", maxSessionQuarterIndex+1)

print(docsFrame[90:100])

totalDocsPerQuarter = []

for i in range(8):
    totalDocsPerQuarter.append(len(docsFrame[docsFrame['Quarter'] == i]))
print(totalDocsPerQuarter) 

N = len(totalDocsPerQuarter)
xvalues = np.arange(N)
xlabels = ['Y1/Q1', 'Y1/Q2', 'Y1/Q3', 'Y1/Q4', 'Y2/Q1', 'Y2/Q2', 'Y2/Q3', 'Y2/Q4']
plt.bar(xvalues, totalDocsPerQuarter, width=1.0, edgecolor="black")
plt.ylabel('Total Actions per Quarter')
plt.xticks(xvalues, xlabels)
plt.show()

totalBillsPerQuarter = []
totalResolutionsPerQuarter = []
isBill = (docsFrame['Type'] == 'hr') | (docsFrame['Type'] == 's')
isResolution = (isBill==False)

for i in range(8):  
    totalBillsPerQuarter.append(len(docsFrame[ (docsFrame['Quarter'] == i) & isBill ])) 
    totalResolutionsPerQuarter.append(len(docsFrame[ (docsFrame['Quarter'] == i) & isResolution ]))
        
totalResolutionsPerQuarter 

N = len(totalDocsPerQuarter)
xvalues = np.arange(N)
xlabels = ['Y1/Q1', 'Y1/Q2', 'Y1/Q3', 'Y1/Q4', 'Y2/Q1', 'Y2/Q2', 'Y2/Q3', 'Y2/Q4']
p1 = plt.bar(xvalues, totalResolutionsPerQuarter, color='g', edgecolor="black", width=1.0)
p2 = plt.bar(xvalues, totalBillsPerQuarter, width=1.0, color='b', edgecolor="black", bottom=totalResolutionsPerQuarter)
plt.ylabel('Total Actions per Quarter')
plt.xticks(xvalues, xlabels)
plt.legend((p1[0], p2[0]),('Resolutions', 'Bills'))
plt.show()

totalDocsPerUniqueQuarter = []
numQuarters = maxSessionQuarterIndex+1
for i in range(0,numQuarters):
    totalDocsPerUniqueQuarter.append(len(docsFrame[docsFrame['SessionQuarterIndex']==i]))
print(totalDocsPerUniqueQuarter) 

# Create label set which marks only the first quarter of each year with the year label
sessionQuarterLabels = []
for i in range(0,numQuarters):
    if ( i % 4 ) == 0:
        year = int((i/4) + 1973)
        sessionQuarterLabels.append(str(year))
    else:
        sessionQuarterLabels.append("")
        
print (sessionQuarterLabels)        

# Set the default figure size to be 15 in by 5 in
from pylab import rcParams
rcParams['figure.figsize'] = 15,5

# Create a function for plotting a topic over time
xlabels = sessionQuarterLabels
xvalues = np.arange(len(sessionQuarterLabels))
yvalues = totalDocsPerUniqueQuarter
    
plt.bar(xvalues, yvalues, width=1.0, edgecolor="black")
plt.title("Total Congressional Actions Per Quarter (1973-2017)")
plt.ylabel('Total Actions per Quarter')
plt.xticks(xvalues, xlabels, rotation=90)
plt.show()

# Load the topic distributions for all documents from file
ldaDocTopicProbsFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicProbs.npy")
docTopicProbs = np.load(ldaDocTopicProbsFile)
docTopicProbs.shape

# Aggregate the topic contributions for each document into topic bins for each quarter
numQuarters = maxSessionQuarterIndex + 1;
numTopics = docTopicProbs.shape[1]
numDocs = len(docsFrame)
topicQuarterRawCounts = np.zeros((numTopics, numQuarters))
for docIndex in range(0,numDocs):
    quarter = docsFrame['SessionQuarterIndex'][docIndex]
    for topicID in range(0,numTopics):
        topicQuarterRawCounts[topicID, quarter] += docTopicProbs[docIndex, topicID]

# Get the topic summaries to use as titles for each topic plot
ldaTopicSummariesFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicSummaries.tsv")
topicSummaries = pandas.read_csv(ldaTopicSummariesFile, sep='\t')

# Set the default figure size to be 15 by 5 in
from pylab import rcParams
rcParams['figure.figsize'] = 15, 5

# Create a function for plotting a topic over time
def PlotTopic(topicID, topicQuarterRawCounts, ylabel, xlabels, topicSummaries):
    xvalues = np.arange(len(xlabels))
    yvalues = topicQuarterRawCounts[topicID]
    
    plt.bar(xvalues, yvalues, width=1.0, edgecolor="black")
    plt.title(topicSummaries['TopicSummary'][topicID])
    plt.ylabel(ylabel)
    plt.xticks(xvalues+0.50, xlabels, rotation=90)
    plt.show()

# Plot topic 165 (which was the top ranked topic identified in Part 4)
PlotTopic(165, topicQuarterRawCounts, 'Total Estimated Actions per Quarter', sessionQuarterLabels, topicSummaries)

# Show a plot of topic 140 which was identified as the sixth highest ranked topic in Part 4.
PlotTopic(140, topicQuarterRawCounts, 'Total Estimated Actions per Quarter', sessionQuarterLabels, topicSummaries)

PlotTopic(38, topicQuarterRawCounts, 'Total Estimated Actions per Quarter', sessionQuarterLabels, topicSummaries)

PlotTopic(168, topicQuarterRawCounts, 'Total Estimated Actions per Quarter', sessionQuarterLabels, topicSummaries)

np.seterr(divide='ignore', invalid='ignore')

# This array contain the probability that a randomly selected document 
# came from a specific quarter for the data time span (1973-2017) 
probQuarter = np.array(totalDocsPerUniqueQuarter, dtype='f') / sum(totalDocsPerUniqueQuarter)

# This array contains the prior probability of a topic across the whole corpus
probTopic = docTopicProbs.sum(axis=0)[:, np.newaxis]
probTopic = probTopic / np.sum(probTopic)

# Compute the conditional probability of a topic given a specific quarter  
normTopicGivenQuarter = (np.sum(topicQuarterRawCounts, axis=0))[:, np.newaxis]
probTopicGivenQuarter = np.transpose(np.transpose(topicQuarterRawCounts) / normTopicGivenQuarter)

# Compute the conditional probability of a specific quarter given a topic
probQuarterGivenTopic = topicQuarterRawCounts / (np.sum(topicQuarterRawCounts,axis=1)[:, np.newaxis])

# Produce a "heat" indicator to highlight quarters for which a topic has higher than expected activity
topicHeatMap = 10000 * probQuarterGivenTopic * probTopicGivenQuarter * np.log((probQuarterGivenTopic / probQuarter))

PlotTopic(165, topicHeatMap, 'Anomalous Activity Score', sessionQuarterLabels, topicSummaries)

PlotTopic(168, topicHeatMap, 'Anomalous Activity Score', sessionQuarterLabels, topicSummaries)

PlotTopic(38, topicHeatMap, 'Anomalous Activity Score', sessionQuarterLabels, topicSummaries)

topicTermProbs = np.load(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicLM.npy"))
topicTermProbs.shape

# Topic Similarity 
# First compute unit normalized vectors
normVector = np.matrix(np.sqrt(np.sum(np.array(topicTermProbs) * np.array(topicTermProbs), axis=1))).transpose()
topicTermProbsUnitNormed = np.matrix(np.array(topicTermProbs) / np.array(normVector))

# Compute topicSimilarity using cosine simlarity measure
topicSimilarity = topicTermProbsUnitNormed * topicTermProbsUnitNormed.transpose()
topicSimilarity.shape

print(topicSimilarity)

def PrintSimilarTopics(topicID, topicSimilarity, topicSummaries, topN):
    sortedTopics = np.array(np.argsort(-topicSimilarity[topicID]))[0]
    for i in range(topN+1):
        print ("%4.3f %3d : %s" % (topicSimilarity[topicID,sortedTopics[i]], 
                                   sortedTopics[i], 
                                   topicSummaries['TopicSummary'][sortedTopics[i]]))

PrintSimilarTopics(38, topicSimilarity, topicSummaries, 10)

PrintSimilarTopics(168, topicSimilarity, topicSummaries, 10)

topicDistances = -np.log2(topicSimilarity)
# For some reason diagonal elements are not exactly zero...so force them to zero
for i in range(0,numTopics):
    topicDistances[i,i]=0

# Extract the upper right diagonal of topicDistances into a condensed 
# distance format for clustering and pass it into the hierarchical 
# clustering algorithm using the max (or 'complete') distance metric
topicClustering=linkage(ssd.squareform(topicDistances), 'complete')

# Plot a dendrogram of the hierarchical clustering of topics
def PlotTopicDendrogram(topicClustering, topicSummaries):    
    numTopics = len(topicSummaries)
    if numTopics != len(topicClustering) + 1:
        print ("Error: Number of topics in topic label set (%d) and topic clustering (%d) are not equal"
               % (numTopics, len(topicClustering) + 1)
              )
        return
    height = int(numTopics/4)
    
    plt.figure(figsize=(10,height))
    plt.title('Topic Dendrogram')
    plt.xlabel('Topical Distance')
    dendrogram(topicClustering, leaf_font_size=12, orientation='right', labels=topicSummaries)
    plt.show()
    return
    
PlotTopicDendrogram(topicClustering,list(topicSummaries['TopicSummary']))

