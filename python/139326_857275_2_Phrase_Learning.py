import pandas 
import re
import math
from operator import itemgetter
from collections import namedtuple
from datetime import datetime
from multiprocessing import cpu_count
from math import log
from sys import getsizeof
import concurrent.futures
import threading
import platform
import time
import gc
import sys
import os

from azureml.logging import get_azureml_logger
aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
aml_logger.log('amlrealworld.document-collection-analysis.notebook2', 'true')

textFrame = pandas.read_csv(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'CongressionalDocsCleaned.tsv'), 
                            sep='\t', 
                            encoding='ISO-8859-1')

print ("Total lines in cleaned text: %d\n" % len(textFrame))

# Show the first 25 rows of the data in the frame
textFrame[0:25]

# Create a lowercased version of the data and add it into the data frame
lowercaseText = []
for textLine in textFrame['CleanedText']:
    lowercaseText.append(str(textLine).lower())
textFrame['LowercaseText'] = lowercaseText;           
            
textFrame[0:25]

# Define a function for loading lists into dictionary hash tables
def LoadListAsHash(filename):
    listHash = {}
    fp = open(filename, encoding='utf-8')

    # Read in lines one by one stripping away extra spaces, 
    # leading spaces, and trailing spaces and inserting each
    # cleaned up line into a hash table
    re1 = re.compile(' +')
    re2 = re.compile('^ +| +$')
    for stringIn in fp.readlines():
        term = re2.sub("",re1.sub(" ",stringIn.strip('\n')))
        if term != '':
            listHash[term] = 1

    fp.close()
    return listHash 

# Load the black list of words to ignore 
blacklistHash = LoadListAsHash(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'black_list.txt'))

# Load the list of non-content bearing function words
functionwordHash = LoadListAsHash(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'function_words.txt'))

# Add more terms to the function word list
functionwordHash["foo"] = 1

# This is the function that used to define how to compute N-gram stats
# This function will be executed in parallel as process pool executor
def ComputeNgramStatsJob(textList, functionwordHash, blacklistHash, reValidWord, jobId, verbose=False):
    if verbose:
        startTS = datetime.now()
        print("[%s] Starting batch execution %d" % (str(startTS), jobId+1))
    
    # Create an array to store the total count of all ngrams up to 4-grams
    # Array element 0 is unused, element 1 is unigrams, element 2 is bigrams, etc.
    ngramCounts = [0]*5;
       
    # Create a list of structures to tabulate ngram count statistics
    # Array element 0 is the array of total ngram counts,
    # Array element 1 is a hash table of individual unigram counts
    # Array element 2 is a hash table of individual bigram counts
    # Array element 3 is a hash table of individual trigram counts
    # Array element 4 is a hash table of individual 4-gram counts
    ngramStats = [ngramCounts, {}, {}, {}, {}]
    
    numLines = len(textList)
    if verbose:
        print("# Batch %d, received %d lines data" % (jobId+1, numLines))
    
    for i in range(0, numLines):
        # Split the text line into an array of words
        wordArray = textList[i].strip().split()
        numWords = len(wordArray)
        
        # Create an array marking each word as valid or invalid
        validArray = [reValidWord.match(word) != None for word in wordArray]
        
        # Tabulate total raw ngrams for this line into counts for each ngram bin
        # The total ngrams counts include the counts of all ngrams including those
        # that we won't consider as parts of phrases
        for j in range(1, 5):
            if j <= numWords:
                ngramCounts[j] += numWords - j + 1
        
        # Collect counts for viable phrase ngrams and left context sub-phrases
        for j in range(0, numWords):
            word = wordArray[j]

            # Only bother counting the ngrams that start with a valid content word
            # i.e., valid words not in the function word list or the black list
            if ( ( word not in functionwordHash ) and ( word not in blacklistHash ) and validArray[j] ):

                # Initialize ngram string with first content word and add it to unigram counts
                ngramSeq = word 
                if ngramSeq in ngramStats[1]:
                    ngramStats[1][ngramSeq] += 1
                else:
                    ngramStats[1][ngramSeq] = 1

                # Count valid ngrams from bigrams up to 4-grams
                stop = 0
                k = 1
                while (k<4) and (j+k<numWords) and not stop:
                    n = k + 1
                    nextNgramWord = wordArray[j+k]
                    # Only count ngrams with valid words not in the blacklist
                    if ( validArray[j+k] and nextNgramWord not in blacklistHash ):
                        ngramSeq += " " + nextNgramWord
                        if ngramSeq in ngramStats[n]:
                            ngramStats[n][ngramSeq] += 1
                        else:
                            ngramStats[n][ngramSeq] = 1 
                        k += 1
                        if nextNgramWord not in functionwordHash:
                            # Stop counting new ngrams after second content word in 
                            # ngram is reached and ngram is a viable full phrase
                            stop = 1
                    else:
                        stop = 1
    
    if verbose:
        endTS = datetime.now()
        delta_t = (endTS - startTS).total_seconds()
        print("[%s] Batch %d finished, time elapsed: %f seconds" % (str(endTS), jobId+1, delta_t))
    
    return ngramStats

# This is Step 1 for each iteration of phrase learning
# We count the number of occurrences of all 2-gram, 3-ngram, and 4-gram
# word sequences 
def ComputeNgramStats(textData, functionwordHash, blacklistHash, numWorkers, verbose=False):
          
    # Create a regular expression for assessing validity of words
    # for phrase modeling. The expression says words in phrases
    # must either:
    # (1) contain an alphabetic character, or 
    # (2) be the single charcater '&', or
    # (3) be a one or two digit number
    reWordIsValid = re.compile('[A-Za-z]|^&$|^\d\d?$');
    
    # Go through the text data line by line collecting count statistics
    # for all valid n-grams that could appear in a potential phrase
    numLines = len(textData)
    
    # Get the number of CPU to run the tasks
    if numWorkers > cpu_count() or numWorkers <= 0:
        worker = cpu_count()
    else:
        worker = numWorkers
    if verbose:
        print("Worker size = %d" % worker)
    
    # Get the batch size for each execution job
    # The very last job executor may received more lines of data
    batch_size = int(numLines/worker)
    batchIndexes = range(0, numLines, batch_size)
    
    batch_returns = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        jobs = set()
        
        # Map the task into multiple batch executions
        if platform.system() == "Linux" or platform.system() == "Darwin":
            for idx in range(worker):
                # The very last job executor
                if idx == (worker-1):
                    jobs.add(executor.submit(ComputeNgramStatsJob, 
                                                 textData[batchIndexes[idx]: ], 
                                                 functionwordHash, 
                                                 blacklistHash,
                                                 reWordIsValid,
                                                 idx, 
                                                 verbose))
                else:
                    jobs.add(executor.submit(ComputeNgramStatsJob, 
                                                 textData[batchIndexes[idx]:(batchIndexes[idx]+batch_size)], 
                                                 functionwordHash, 
                                                 blacklistHash,
                                                 reWordIsValid,
                                                 idx,
                                                 verbose))
        else:
            # For Windows system, it is different to handle ProcessPoolExecutor
            from notebooks import winprocess
            
            for idx in range(worker):
                # The very last job executor
                if idx == (worker-1):
                    jobs.add(winprocess.submit(executor,
                                                 ComputeNgramStatsJob, 
                                                 textData[batchIndexes[idx]: ], 
                                                 functionwordHash, 
                                                 blacklistHash,
                                                 reWordIsValid,
                                                 idx, 
                                                 verbose))
                else:
                    jobs.add(winprocess.submit(executor,
                                                 ComputeNgramStatsJob, 
                                                 textData[batchIndexes[idx]:(batchIndexes[idx]+batch_size)], 
                                                 functionwordHash, 
                                                 blacklistHash,
                                                 reWordIsValid,
                                                 idx,
                                                 verbose))
        
        # Get results from batch executions
        for job in concurrent.futures.as_completed(jobs):
            try:
                ret = job.result()
            except Exception as e:
                print("Generated an exception while trying to get result from a batch: %s" % e)
            else:
                batch_returns.append(ret)

    # Reduce the results from batch executions
    # Reuse the first return
    ngramStats = batch_returns[0]
    
    for batch_id in range(1, len(batch_returns)):
        result = batch_returns[batch_id]
        
        # Update the ngram counts
        ngramStats[0] = [x + y for x, y in zip(ngramStats[0], result[0])]
        
        # Update the hash table of ngram counts
        for n_gram in range(1, 5):
            for item in result[n_gram]:
                if item in ngramStats[n_gram]:
                    ngramStats[n_gram][item] += result[n_gram][item]
                else:
                    ngramStats[n_gram][item] = result[n_gram][item]
    
    return ngramStats

def RankNgrams(ngramStats,functionwordHash,minCount):
    # Create a hash table to store weighted pointwise mutual 
    # information scores for each viable phrase
    ngramWPMIHash = {}
        
    # Go through each of the ngram tables and compute the phrase scores
    # for the viable phrases
    for n in range(2,5):
        i = n-1
        for ngram in ngramStats[n].keys():
            ngramCount = ngramStats[n][ngram]
            if ngramCount >= minCount:
                wordArray = ngram.split()
                # If the final word in the ngram is not a function word then
                # the ngram is a valid phrase candidate we want to score
                if wordArray[i] not in functionwordHash: 
                    leftNgram = ' '.join(wordArray[:-1])
                    rightWord = wordArray[i]
                    
                    # Compute the weighted pointwise mutual information (WPMI) for the phrase
                    probNgram = float(ngramStats[n][ngram])/float(ngramStats[0][n])
                    probLeftNgram = float(ngramStats[n-1][leftNgram])/float(ngramStats[0][n-1])
                    probRightWord = float(ngramStats[1][rightWord])/float(ngramStats[0][1])
                    WPMI = probNgram * math.log(probNgram/(probLeftNgram*probRightWord));

                    # Add the phrase into the list of scored phrases only if WMPI is positive
                    if WPMI > 0:
                        ngramWPMIHash[ngram] = WPMI  
    
    # Create a sorted list of the phrase candidates
    rankedNgrams = sorted(ngramWPMIHash, key=ngramWPMIHash.__getitem__, reverse=True)

    # Force a memory clean-up
    ngramWPMIHash = None
    gc.collect()

    return rankedNgrams

def phraseRewriteJob(ngramRegex, text, ngramRewriteHash, jobId, verbose=True):
    if verbose:
        startTS = datetime.now()
        print("[%s] Starting batch execution %d" % (str(startTS), jobId+1))
    
    retList = []
    
    for i in range(len(text)):
        # The regex substitution looks up the output string rewrite
        # in the hash table for each matched input phrase regex
        retList.append(ngramRegex.sub(lambda mo: ngramRewriteHash[mo.string[mo.start():mo.end()]], text[i]))
    
    if verbose:
        endTS = datetime.now()
        delta_t = (endTS - startTS).total_seconds()
        print("[%s] Batch %d finished, batch size: %d, time elapsed: %f seconds" % (str(endTS), jobId+1, i, delta_t))
    
    return retList, jobId

def ApplyPhraseRewrites(rankedNgrams, textData, learnedPhrases, maxPhrasesToAdd, 
                        maxPhraseLength, verbose, numWorkers=cpu_count()):

    # If the number of rankedNgrams coming in is zero then
    # just return without doing anything
    numNgrams = len(rankedNgrams)
    if numNgrams == 0:
        return

    # This function will consider at most maxRewrite 
    # new phrases to be added into the learned phrase 
    # list as specified by the calling function
    maxRewrite=maxPhrasesToAdd

    # If the remaining number of proposed ngram phrases is less 
    # than the max allowed, then reset maxRewrite to the size of 
    # the proposed ngram phrases list
    if numNgrams < maxRewrite:
        maxRewrite = numNgrams

    # Create empty hash tables to keep track of phrase overlap conflicts
    leftConflictHash = {}
    rightConflictHash = {}
    
    # Create an empty hash table collecting the set of rewrite rules
    # to be applied during this iteration of phrase learning
    ngramRewriteHash = {}
    
    # Precompile the regex for finding spaces in ngram phrases
    regexSpace = re.compile(' ')

    # Initialize some bookkeeping variables
    numLines = len(textData)  
    numPhrasesAdded = 0
    numConsidered = 0
    lastSkippedNgram = ""
    lastAddedNgram = ""
  
    # Collect list of up to maxRewrite ngram phrase rewrites
    stop = False
    index = 0
    while not stop:

        # Get the next phrase to consider adding to the phrase list
        inputNgram = rankedNgrams[index]

        # Create the output compound word version of the phrase
        # The extra space is added to make the regex rewrite easier
        outputNgram = " " + regexSpace.sub("_",inputNgram)

        # Count the total number of words in the proposed phrase
        numWords = len(outputNgram.split("_"))

        # Only add phrases that don't exceed the max phrase length
        if (numWords <= maxPhraseLength):
    
            # Keep count of phrases considered for inclusion during this iteration
            numConsidered += 1

            # Extract the left and right words in the phrase to use
            # in checks for phrase overlap conflicts
            ngramArray = inputNgram.split()
            leftWord = ngramArray[0]
            rightWord = ngramArray[-1]

            # Skip any ngram phrases that conflict with earlier phrases added
            # These ngram phrases will be reconsidered in the next iteration
            if (leftWord in leftConflictHash) or (rightWord in rightConflictHash): 
                if verbose: 
                    print ("(%d) Skipping (context conflict): %s" % (numConsidered,inputNgram))
                lastSkippedNgram = inputNgram
                
            # If no conflict exists then add this phrase into the list of phrase rewrites     
            else: 
                if verbose:
                    print ("(%d) Adding: %s" % (numConsidered,inputNgram))
                ngramRewriteHash[" " + inputNgram] = outputNgram
                learnedPhrases.append(inputNgram) 
                lastAddedNgram = inputNgram
                numPhrasesAdded += 1
            
            # Keep track of all context words that might conflict with upcoming
            # propose phrases (even when phrases are skipped instead of added)
            leftConflictHash[rightWord] = 1
            rightConflictHash[leftWord] = 1

            # Stop when we've considered the maximum number of phrases per iteration
            if ( numConsidered >= maxRewrite ):
                stop = True
            
        # Increment to next phrase
        index += 1
    
        # Stop if we've reached the end of the ranked ngram list
        if index >= len(rankedNgrams):
            stop = True
    
    # Now do the phrase rewrites over the entire set of text data
    # Compile a single regex rule from the collected set of phrase rewrites for this iteration
    ngramRegex = re.compile(r'%s(?= )' % "(?= )|".join(map(re.escape, ngramRewriteHash.keys())))
    
    # Get the number of CPU to run the tasks
    if numWorkers > cpu_count() or numWorkers <= 0:
        worker = cpu_count()
    else:
        worker = numWorkers
    if verbose:
        print("Worker size = %d" % worker)
        
    # Get the batch size for each execution job
    # The very last job executor may receive more lines of data
    batch_size = int(numLines/worker)
    batchIndexes = range(0, numLines, batch_size)
    
    batch_returns = [[]] * worker
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        jobs = set()
        
        # Map the task into multiple batch executions
        if platform.system() == "Linux" or platform.system() == "Darwin":
            for idx in range(worker):
                if idx == (worker-1):
                    jobs.add(executor.submit(phraseRewriteJob, 
                                             ngramRegex, 
                                             textData[batchIndexes[idx]: ], 
                                             ngramRewriteHash, 
                                             idx,
                                             verbose))
                else:
                    jobs.add(executor.submit(phraseRewriteJob, 
                                             ngramRegex, 
                                             textData[batchIndexes[idx]:(batchIndexes[idx]+batch_size)], 
                                             ngramRewriteHash, 
                                             idx,
                                             verbose))
        else:
            from notebooks import winprocess
            
            for idx in range(worker):
                if idx == (worker-1):
                    jobs.add(winprocess.submit(executor,
                                             phraseRewriteJob, 
                                             ngramRegex, 
                                             textData[batchIndexes[idx]: ], 
                                             ngramRewriteHash, 
                                             idx,
                                             verbose))
                else:
                    jobs.add(winprocess.submit(executor,
                                             phraseRewriteJob, 
                                             ngramRegex, 
                                             textData[batchIndexes[idx]:(batchIndexes[idx]+batch_size)], 
                                             ngramRewriteHash, 
                                             idx,
                                             verbose))
        
        textData.clear()
        
        # Get results from batch executions
        for job in concurrent.futures.as_completed(jobs):
            try:
                ret, idx = job.result()
            except Exception as e:
                print("Generated an exception while trying to get result from a batch: %s" % e)
            else:
                batch_returns[idx] = ret
        textData += sum(batch_returns, [])
     
    return

def ApplyPhraseLearning(textData,learnedPhrases,learningSettings):
    
    stop = False
    iterNum = 0

    # Get the learning parameters from the structure passed in by the calling function
    maxNumPhrases = learningSettings.maxNumPhrases
    maxPhraseLength = learningSettings.maxPhraseLength
    functionwordHash = learningSettings.functionwordHash
    blacklistHash = learningSettings.blacklistHash
    verbose = learningSettings.verbose
    minCount = learningSettings.minInstanceCount
    
    # Start timing the process
    functionStartTime = time.clock()
    
    numPhrasesLearned = len(learnedPhrases)
    print ("Start phrase learning with %d phrases of %d phrases learned" % (numPhrasesLearned,maxNumPhrases))

    while not stop:
        iterNum += 1
                
        # Start timing this iteration
        startTime = time.clock()
 
        # Collect ngram stats
        ngramStats = ComputeNgramStats(textData, functionwordHash, blacklistHash, cpu_count(), verbose)

        # Uncomment this for more detailed timing info
        countTime = time.clock()
        elapsedTime = countTime - startTime
        print ("--- Counting time: %.2f seconds" % elapsedTime)
        
        # Rank ngrams
        rankedNgrams = RankNgrams(ngramStats,functionwordHash,minCount)
        
        # Uncomment this for more detailed timing info
        rankTime = time.clock()
        elapsedTime = rankTime - countTime
        print ("--- Ranking time: %.2f seconds" % elapsedTime)
        
        
        # Incorporate top ranked phrases into phrase list
        # and rewrite the text to use these phrases
        if len(rankedNgrams) > 0:
            maxPhrasesToAdd = maxNumPhrases - numPhrasesLearned
            if maxPhrasesToAdd > learningSettings.maxPhrasesPerIter:
                maxPhrasesToAdd = learningSettings.maxPhrasesPerIter
            ApplyPhraseRewrites(rankedNgrams, textData, learnedPhrases, maxPhrasesToAdd, 
                                maxPhraseLength, verbose, cpu_count())
            numPhrasesAdded = len(learnedPhrases) - numPhrasesLearned
        else:
            stop = True
            
        # Uncomment this for more detailed timing info
        rewriteTime = time.clock()
        elapsedTime = rewriteTime - rankTime
        print ("--- Rewriting time: %.2f seconds" % elapsedTime)
           
        # Garbage collect
        ngramStats = None
        rankedNgrams = None
        gc.collect();
               
        elapsedTime = time.clock() - startTime

        numPhrasesLearned = len(learnedPhrases)
        print ("Iteration %d: Added %d new phrases in %.2f seconds (Learned %d of max %d)" % 
               (iterNum,numPhrasesAdded,elapsedTime,numPhrasesLearned,maxNumPhrases))
        
        if numPhrasesAdded >= maxPhrasesToAdd or numPhrasesAdded == 0:
            stop = True
        
    # Remove the space padding at the start and end of each line
    regexSpacePadding = re.compile('^ +| +$')
    for i in range(0,len(textData)):
        textData[i] = regexSpacePadding.sub("",textData[i])
    
    gc.collect()
 
    elapsedTime = time.clock() - functionStartTime
    elapsedTimeHours = elapsedTime/3600.0;
    print ("*** Phrase learning completed in %.2f hours ***" % elapsedTimeHours) 

    return

# Create a structure defining the settings and word lists used during the phrase learning
learningSettings = namedtuple('learningSettings',['maxNumPhrases','maxPhrasesPerIter',
                                                  'maxPhraseLength','minInstanceCount'
                                                  'functionwordHash','blacklistHash','verbose'])

# If true it prints out the learned phrases to stdout buffer
# while its learning. This will generate a lot of text to stdout, 
# so best to turn this off except for testing and debugging
learningSettings.verbose = False

# Maximum number of phrases to learn
# If you want to test the code out quickly then set this to a small
# value (e.g. 100) and set verbose to true when running the quick test
learningSettings.maxNumPhrases = 25000

# Maximum number of phrases to learn per iteration 
# Increasing this number may speed up processing but will affect the ordering of the phrases 
# learned and good phrases could be by-passed if the maxNumPhrases is set to a small number
learningSettings.maxPhrasesPerIter = 500

# Maximum number of words allowed in the learned phrases 
learningSettings.maxPhraseLength = 7

# Minimum number of times a phrase must occur in the data to 
# be considered during the phrase learning process
learningSettings.minInstanceCount = 5

# This is a precreated hash table containing the list 
# of function words used during phrase learning
learningSettings.functionwordHash = functionwordHash

# This is a precreated hash table containing the list 
# of black list words to be ignored during phrase learning
learningSettings.blacklistHash = blacklistHash

# Initialize an empty list of learned phrases
# If you have completed a partial run of phrase learning
# and want to add more phrases, you can use the pre-learned 
# phrases as a starting point instead and the new phrases
# will be appended to the list
learnedPhrases = []

# Create a copy of the original text data that will be used during learning
# The copy is needed because the algorithm does in-place replacement of learned
# phrases directly on the text data structure it is provided
phraseTextData = []
for textLine in textFrame['LowercaseText']:
    phraseTextData.append(' ' + textLine + ' ')

# Run the phrase learning algorithm
if True:
    ApplyPhraseLearning(phraseTextData, learnedPhrases, learningSettings)

learnedPhrasesFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocsLearnedPhrases.txt")
phraseTextDataFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "CongressionalDocsPhraseTextData.txt")

writeLearnedPhrases = True

if writeLearnedPhrases:
    # Write out the learned phrases to a text file
    fp = open(learnedPhrasesFile, 'w', encoding='utf-8')
    for phrase in learnedPhrases:
        fp.write("%s\n" % phrase)
    fp.close()

    # Write out the text data containing the learned phrases to a text file
    fp = open(phraseTextDataFile, 'w', encoding='utf-8')
    for line in phraseTextData:
        fp.write("%s\n" % line)
    fp.close()
else:
    # Read in the learned phrases from a text file
    learnedPhrases = []
    fp = open(learnedPhrasesFile, 'r', encoding='utf-8')
    for line in fp:
        learnedPhrases.append(line.strip())
    fp.close()

    # Read in the learned phrases from a text file
    phraseTextData = []
    fp = open(phraseTextDataFile, 'r', encoding='utf-8')
    for line in fp:
        phraseTextData.append(line.strip())
    fp.close()

learnedPhrases[0:10]

learnedPhrases[5000:5010]

phraseTextData[0:15]

# Add text with learned phrases back into data frame
textFrame['TextWithPhrases'] = phraseTextData

textFrame[0:10]

textFrame['TextWithPhrases'][2]

def MapVocabToSurfaceForms(textData):
    surfaceFormCountHash = {}
    vocabToSurfaceFormHash = {}
    regexUnderBar = re.compile('_')
    regexSpace = re.compile(' +')
    regexClean = re.compile('^ +| +$')
    
    # First go through every line of text, align each word/phrase with
    # it's surface form and count the number of times each surface form occurs
    for i in range(0,len(textData)):    
        origWords = regexSpace.split(regexClean.sub("",str(textData['CleanedText'][i])))
        numOrigWords = len(origWords)
        newWords = regexSpace.split(regexClean.sub("",str(textData['TextWithPhrases'][i])))
        numNewWords = len(newWords)
        origIndex = 0
        newIndex = 0
        while newIndex < numNewWords:
            # Get the next word or phrase in the lower-cased text with phrases and
            # match it to the original form of the same n-gram in the original text
            newWord = newWords[newIndex]
            phraseWords = regexUnderBar.split(newWord)
            numPhraseWords = len(phraseWords)
            matchedWords = " ".join(origWords[origIndex:(origIndex+numPhraseWords)])
            origIndex += numPhraseWords
                
            # Now do the bookkeeping for collecting the different surface form 
            # variations present for each lowercased word or phrase
            if newWord in vocabToSurfaceFormHash:
                vocabToSurfaceFormHash[newWord].add(matchedWords)
            else:
                vocabToSurfaceFormHash[newWord] = set([matchedWords])

            # Increment the counter for this surface form
            if matchedWords not in surfaceFormCountHash:
                surfaceFormCountHash[matchedWords] = 1
            else:
                surfaceFormCountHash[matchedWords] += 1
   
            if ( len(newWord) != len(matchedWords)):
                print ("##### Error #####")
                print ("Bad Match: %s ==> %s " % (newWord,matchedWords))
                print ("From line: %s" % textData['TextWithPhrases'][i])
                print ("Orig text: %s" % textData['CleanedText'][i])
                
                return False

            newIndex += 1
    # After aligning and counting, select the most common surface form for each

    # word/phrase to be the canonical example shown to the user for that word/phrase
    for ngram in vocabToSurfaceFormHash.keys():
        maxCount = 0
        bestSurfaceForm = ""
        for surfaceForm in vocabToSurfaceFormHash[ngram]:
            if surfaceFormCountHash[surfaceForm] > maxCount:
                maxCount = surfaceFormCountHash[surfaceForm]
                bestSurfaceForm = surfaceForm
        if ngram != "":
            if bestSurfaceForm == "":
                print ("Warning: NULL surface form for ngram '%s'" % ngram)
            else:
                vocabToSurfaceFormHash[ngram] = bestSurfaceForm
    
    return vocabToSurfaceFormHash

get_ipython().run_cell_magic('time', '', '\nif True:\n    vocabToSurfaceFormHash = MapVocabToSurfaceForms(textFrame)')

# Save the mapping between model vocabulary and surface form mapping
tsvFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], "Vocab2SurfaceFormMapping.tsv")

saveSurfaceFormFile = True

if saveSurfaceFormFile:
    with open(tsvFile, 'w', encoding='utf-8') as fp:
        for key, val in vocabToSurfaceFormHash.items():
            if key != "":
                strOut = "%s\t%s\n" % (key, val)
                fp.write(strOut)
else:
    # Load surface form mappings here
    vocabToSurfaceFormHash = {}
    fp = open(tsvFile, encoding='utf-8')

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

print(vocabToSurfaceFormHash['security'])
print(vocabToSurfaceFormHash['declares'])
print(vocabToSurfaceFormHash['mental_health'])
print(vocabToSurfaceFormHash['el_salvador'])
print(vocabToSurfaceFormHash['department_of_the_interior'])

def ReconstituteDocsFromChunks(textData, idColumnName, textColumnName):
    dataOut = []
    
    currentDoc = ""
    currentDocID = ""
    
    for i in range(0,len(textData)):
        textChunk = textData[textColumnName][i]
        docID = str(textData[idColumnName][i])
        if docID != currentDocID:
            if currentDocID != "":
                dataOut.append([currentDocID, currentDoc])
            currentDoc = textChunk
            currentDocID = docID
        else:
            currentDoc += " " + textChunk
    dataOut.append([currentDocID,currentDoc])
    
    frameOut = pandas.DataFrame(dataOut, columns=['DocID','ProcessedText'])
    
    return frameOut

get_ipython().run_cell_magic('time', '', "\nif True:\n    docsFrame = ReconstituteDocsFromChunks(textFrame, 'DocID', 'TextWithPhrases')")

saveProcessedText = True

# Save processed text for each document back out to a TSV file
if saveProcessedText:
    docsFrame.to_csv(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'CongressionalDocsProcessed.tsv'),  
                        sep='\t', index=False)
else: 
    docsFrame = pandas.read_csv(os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'CongressionalDocsProcessed.tsv'), 
                                    sep='\t')

docsFrame[0:5]

docsFrame['ProcessedText'][1]

def ApplyPhraseRewritesInPlace(textFrame, textColumnName, phraseRules):
    
    # Make sure we have phrase to add
    numPhraseRules = len(phraseRules)
    if numPhraseRules == 0: 
        print ("Warning: phrase rule lise is empty - no phrases being applied to text data")
        return
    
    # Get text data column from frame
    textData = textFrame[textColumnName]
    numLines = len(textData)
    
    # Add leading and trailing spaces to make regex matching easier
    for i in range(0,numLines):
        textData[i] = " " + textData[i] + " "  

    # Precompile the regex for finding spaces in ngram phrases
    regexSpace = re.compile(' ')
   
    # Initialize some bookkeeping variables

    # Iterate through full set of phrases to find sets of 
    # non-conflicting phrases that can be apply simultaneously
    index = 0
    outerStop = False
    while not outerStop:
       
        # Create empty hash tables to keep track of phrase overlap conflicts
        leftConflictHash = {}
        rightConflictHash = {}
        prevConflictHash = {}
    
        # Create an empty hash table collecting the next set of rewrite rules
        # to be applied during this iteration of phrase rewriting
        phraseRewriteHash = {}
    
        # Progress through phrases until the next conflicting phrase is found
        innerStop = 0
        numPhrasesAdded = 0
        while not innerStop:
        
            # Get the next phrase to consider adding to the phrase list
            nextPhrase = phraseRules[index]            
            
            # Extract the left and right sides of the phrase to use
            # in checks for phrase overlap conflicts
            ngramArray = nextPhrase.split()
            leftWord = ngramArray[0]
            rightWord = ngramArray[-1] 

            # Stop if we reach any phrases that conflicts with earlier phrases in this iteration
            # These ngram phrases will be reconsidered in the next iteration
            if ((leftWord in leftConflictHash) or (rightWord in rightConflictHash) 
                or (leftWord in prevConflictHash) or (rightWord in prevConflictHash)): 
                innerStop = True
                
            # If no conflict exists then add this phrase into the list of phrase rewrites     
            else: 
                # Create the output compound word version of the phrase
                                
                outputPhrase = regexSpace.sub("_",nextPhrase);
                
                # Keep track of all context words that might conflict with upcoming
                # propose phrases (even when phrases are skipped instead of added)
                leftConflictHash[rightWord] = 1
                rightConflictHash[leftWord] = 1
                prevConflictHash[outputPhrase] = 1           
                
                # Add extra space to input an output versions of the current phrase 
                # to make the regex rewrite easier
                outputPhrase = " " + outputPhrase
                lastAddedPhrase = " " + nextPhrase
                
                # Add the phrase to the rewrite hash
                phraseRewriteHash[lastAddedPhrase] = outputPhrase
                  
                # Increment to next phrase
                index += 1
                numPhrasesAdded  += 1
    
                # Stop if we've reached the end of the phrases list
                if index >= numPhraseRules:
                    innerStop = True
                    outerStop = True
                    
        # Now do the phrase rewrites over the entire set of text data
        if numPhrasesAdded == 1:
        
            # If only one phrase to add use a single regex rule to do this phrase rewrite        
            outputPhrase = phraseRewriteHash[lastAddedPhrase]
            regexPhrase = re.compile (r'%s(?= )' % re.escape(lastAddedPhrase)) 
        
            # Apply the regex over the full data set
            for j in range(0,numLines):
                textData[j] = regexPhrase.sub(outputPhrase, textData[j])
        
        elif numPhrasesAdded > 1:
            # Compile a single regex rule from the collected set of phrase rewrites for this iteration
            regexPhrase = re.compile(r'%s(?= )' % "|".join(map(re.escape, phraseRewriteHash.keys())))
            
            # Apply the regex over the full data set
            for i in range(0,numLines):
                # The regex substituion looks up the output string rewrite  
                # in the hash table for each matched input phrase regex
                textData[i] = regexPhrase.sub(lambda mo: phraseRewriteHash[mo.string[mo.start():mo.end()]], textData[i]) 
    
    # Remove the space padding at the start and end of each line
    regexSpacePadding = re.compile('^ +| +$')
    for i in range(0,len(textData)):
        textData[i] = regexSpacePadding.sub("",textData[i])
    
    return

testText = ["the president of the united states appoints the secretary of labor to lead the department of labor", 
            "the speaker of the house of representatives is elected each session by the members of the house",
            "the president pro tempore of the the u.s. senate resides over the senate when the vice president is absent"]

testFrame = pandas.DataFrame(testText, columns=['TestText'])      

ApplyPhraseRewritesInPlace(testFrame, 'TestText', learnedPhrases)

print(testFrame['TestText'][0])
print(testFrame['TestText'][1])
print(testFrame['TestText'][2])

