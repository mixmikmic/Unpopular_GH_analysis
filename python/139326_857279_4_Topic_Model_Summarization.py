saveFile = True

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

# Set the saveFile flag to False since you have already downloaded those files
saveFile = False

import numpy
import pandas 
import re
import math
import os
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim import corpora
from gensim import models
from operator import itemgetter
from collections import namedtuple
import time
import gc
import sys
import multiprocessing
import matplotlib
matplotlib.use('Agg')

from azureml.logging import get_azureml_logger
aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
aml_logger.log('amlrealworld.document-collection-analysis.notebook4', 'true')

# Load pretrained LDA topic model
ldaFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocsLDA.pickle")
lda = gensim.models.ldamodel.LdaModel.load(ldaFile)

# Get the mapping from token ID to token string
id2token = lda.id2word
print(id2token[1])

# Load surface form mappings here
fp = open(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "Vocab2SurfaceFormMapping.tsv"), encoding='utf-8')

vocabToSurfaceFormHash = {}

# Each line in the file has two tab separated fields;
# the first is the vocabulary item used during modeling
# and the second is its most common surface form in the 
# original data
for stringIn in fp.readlines():
    fields = stringIn.strip().split("\t")
    if len(fields) != 2:
        print ("Warning: Bad line in surface form mapping file: %s" % stringIn)
    elif fields[0] == "" or fields[1] == "":
        print ("Warning: Bad line in surface form mapping file: %s" % stringIn)
    else:
        vocabToSurfaceFormHash[fields[0]] = fields[1]
fp.close()

def CreateTermIDToSurfaceFormMapping(id2token, token2surfaceform):
    termIDToSurfaceFormMap = []
    for i in range(0, len(id2token)):
        if id2token[i] in token2surfaceform:
            termIDToSurfaceFormMap.append(token2surfaceform[id2token[i]])
    return termIDToSurfaceFormMap;

termIDToSurfaceFormMap = CreateTermIDToSurfaceFormMapping(id2token, vocabToSurfaceFormHash);

# print out the modeled token form and the best matching surface for the token with the index value of 18
i = 18
print('Term index:', i)
print('Modeled form:', id2token[i])
print('Surface form:', termIDToSurfaceFormMap[i])

numTopics = lda.num_topics
print ("Number of topics:", numTopics)

lda.print_topics(10)

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def _terms_to_counts(terms, multiplier=1000):
    return ' '.join([' '.join(int(multiplier * x[1]) * [x[0]]) for x in terms])


def visualizeTopic(lda, topicID=0, topn=500, multiplier=1000):
    terms = []
    tmp = lda.show_topic(topicID, topn)
    for term in tmp:
        terms.append(term)
    
    # If the version of wordcloud is higher than 1.3, then you will need to set 'collocations' to False.
    # Otherwise there will be word duplicates in the figure. 
    try:
        wordcloud = WordCloud(max_words=10000, collocations=False).generate(_terms_to_counts(terms, multiplier))
    except:
        wordcloud = WordCloud(max_words=10000).generate(_terms_to_counts(terms, multiplier))
    fig = plt.figure(figsize=(12, 16))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Topic %d" % topicID)
    plt.show()
    

get_ipython().run_line_magic('matplotlib', 'inline')

visualizeTopic(lda, topicID=38, topn=1000)

visualizeTopic(lda, topicID=168, topn=1000)

docTopicProbsFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicProbs.npy")

# docTopicProbs[docID,TopicID] --> P(topic|doc)
docTopicProbs = numpy.load(docTopicProbsFile)

# The docTopicProbs shape should be (# of docs, # of topics)
docTopicProbs.shape

# Computing the global topic likelihoods by aggregating topic probabilities over all documents
# topicProbs[topicID] --> P(topic)
def ComputeTopicProbs(docTopicProbs):
    topicProbs = docTopicProbs.sum(axis=0) 
    topicProbs = topicProbs/sum(topicProbs)
    return topicProbs

topicProbs = ComputeTopicProbs(docTopicProbs)

def ExtractTopicLMMatrix(lda):
    # Initialize the matrix
    docTopicProbs = numpy.zeros((lda.num_topics,lda.num_terms))
    for topicID in range(0,lda.num_topics):
        termProbsList = lda.get_topic_terms(topicID,lda.num_terms)
        for termProb in termProbsList:
            docTopicProbs[topicID,termProb[0]]=termProb[1]
    return docTopicProbs
    
# topicTermProbs[topicID,termID] --> P(term|topic)
topicTermProbs = ExtractTopicLMMatrix(lda)

# Set saveFile flag to true if you want to save the Topic LMs for a newly trained LDA model to file
if saveFile:
    numpy.save(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicLM.npy"), topicTermProbs)

# Compute the joint likelihoods of topics and terms
# jointTopicTermProbs[topicID,termID] --> P(topic,term) = P(term|topic)*P(topic)
jointTopicTermProbs = numpy.diag(topicProbs).dot(topicTermProbs) 

# termProbs[termID] --> P(term)
termProbs = jointTopicTermProbs.sum(axis=0)

# topicProbsPermTerm[topicID,termID] --> P(topic|term)
topicProbsPerTerm = jointTopicTermProbs / termProbs

# Print most frequent words in LDA vocab
mostFrequentTermIDs = (-termProbs).argsort()
for i in range(0,25):
    print ("%d: %s --> %f" % (i+1, id2token[mostFrequentTermIDs[i]], termProbs[mostFrequentTermIDs[i]]))

topicTermWPMI =(jointTopicTermProbs.T * numpy.log(topicProbsPerTerm.T / topicProbs)).T
topicTermWPMI.shape

topicPurity = numpy.exp(((docTopicProbs * numpy.log(docTopicProbs)).sum(axis=0))/(docTopicProbs).sum(axis=0))

topicID = 38

highestWPMITermIDs = (-topicTermWPMI[topicID]).argsort()
highestProbTermIDs = (-topicTermProbs[topicID]).argsort()
print ("                                        WPMI                                                 Prob")
for i in range(0,15):
    print ("%2d: %35s ---> %8.6f    %35s ---> %8.6f" % (i+1, 
                                                        termIDToSurfaceFormMap[highestWPMITermIDs[i]], 
                                                        topicTermWPMI[topicID,highestWPMITermIDs[i]],
                                                        termIDToSurfaceFormMap[highestProbTermIDs[i]], 
                                                        topicTermProbs[topicID,highestProbTermIDs[i]]))                

def CreateTopicSummaries(topicTermScores, id2token, tokenid2surfaceform, maxStringLen):
    reIgnore = re.compile('^[a-z]\.$')
    reAcronym = re.compile('^[A-Z]+$')
    topicSummaries = []
    for topicID in range(0,len(topicTermScores)):
        rankedTermIDs = (-topicTermScores[topicID]).argsort()
        maxNumTerms = len(rankedTermIDs)
        termIndex = 0
        stop = 0
        outputTokens = []
        prevAcronyms = []
        topicSummary = ""
        while not stop:
            # If we've run out of tokens then stop...
            if (termIndex>=maxNumTerms):
                stop=1
            # ...otherwise consider adding next token to summary
            else:
                nextToken = id2token[rankedTermIDs[termIndex]]
                nextTokenOut = tokenid2surfaceform[rankedTermIDs[termIndex]]
                keepToken = 1
                
                # Prepare to test current word as an acronym or a string that reduces to an acronym
                nextTokenIsAcronym = 0
                nextTokenAbbrev = ""
                if reAcronym.match(nextTokenOut) != None:
                    nextTokenIsAcronym = 1
                else:
                    subTokens = nextToken.split('_')
                    if (len(subTokens)>1):
                        for subToken in subTokens:
                            nextTokenAbbrev += subToken[0]                        

                # See if we should ignore this token because it matches the regex for tokens to ignore
                if ( reIgnore.match(nextToken) != None ):
                    keepToken = 0;

                # Otherwise see if we should ignore this token because
                # it is a close match to a previously selected token
                elif len(outputTokens) > 0:          
                    for prevToken in outputTokens:
                        # Ignore token if it is a substring of a previous token
                        if nextToken in prevToken:
                            keepToken = 0
                        # Ignore token if it is a superstring of a previous token
                        elif prevToken in nextToken:
                            keepToken = 0
                        # Ignore token if it is an acronym of a previous token
                        elif nextTokenIsAcronym:
                            subTokens = prevToken.split('_')
                            if (len(subTokens)>1):
                                prevTokenAbbrev = ""
                                for subToken in subTokens:
                                    prevTokenAbbrev += subToken[0]
                                if prevTokenAbbrev == nextToken:
                                    keepToken = 0                                  
                    for prevAcronym in prevAcronyms:
                        # Ignore token if it is the long form of an earlier acronym
                        if nextTokenAbbrev == prevAcronym:
                                keepToken = 0

                # Add tokens to the summary for this topic                
                if keepToken:
                    # Always add at least one token to the summary
                    if len(topicSummary) == 0 or ( len(topicSummary) + len(nextTokenOut) + 1 < maxStringLen):
                        if len(topicSummary) == 0:
                            topicSummary = nextTokenOut
                        else: 
                            topicSummary += ", " + nextTokenOut
                        outputTokens.append(nextToken)
                        if nextTokenIsAcronym:
                            prevAcronyms.append(nextToken)
                    # If we didn't add the previous word and we're within 10 characters of 
                    # the max string length then we'll just stop here
                    elif maxStringLen - len(topicSummary) < 10 :
                        stop = 1
                    # Otherwise if the current token is too long, but we still have more than
                    # 10 characters of space left we'll just skip this one and add the next token
                    # one if it's short enough
                termIndex += 1         
        topicSummaries.append(topicSummary)
    return topicSummaries   
    
topicSummaries = CreateTopicSummaries(topicTermWPMI, id2token, termIDToSurfaceFormMap, 85)

# Rank the topics by their prominence score in the corpus
# The topic score combines the total weight of each a topic in the corpus 
# with a topic document purity score for topic 
# Topics with topicScore > 1 are generally very strong topics

topicScore = (numTopics * topicProbs) * (2 * topicPurity)
topicRanking = (-topicScore).argsort()

print ("Rank  ID  Score  Prob  Purity  Summary")
for i in range(0, numTopics):
    topicID = topicRanking[i]
    print (" %3d %3d %6.3f (%5.3f, %4.3f) %s" 
           % (i+1, topicID, topicScore[topicID], 100*topicProbs[topicID], topicPurity[topicID], topicSummaries[topicID]))

# If you want to save out the summaries to file makes saveFile flag True
if saveFile:
    fp = open(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocTopicSummaries.tsv"), "w")
    i = 0
    fp.write("TopicID\tTopicSummary\n")
    for line in topicSummaries:
        fp.write("%d\t%s\n" % (i, line))
        i += 1
    fp.close()

