# !pip install --upgrade notebook
# !pip install --upgrade nltk

import pandas as pd
import numpy as np
import re, os, requests, warnings
from collections import (namedtuple, Counter)
from modules.phrase_learning import (CleanAndSplitText, ComputeNgramStats, RankNgrams, ApplyPhraseRewrites,
                            ApplyPhraseLearning, ApplyPhraseRewritesInPlace, ReconstituteDocsFromChunks,
                            CreateVocabForTopicModeling)
from azureml.logging import get_azureml_logger
warnings.filterwarnings("ignore")

run_logger = get_azureml_logger()
run_logger.log('amlrealworld.QnA-matching.part2-phrase-learning','true')

# load non-content bearing function words (.txt file) into a Python dictionary. 
def LoadListAsHash(fileURL):
    response = requests.get(fileURL, stream=True)
    wordsList = response.text.split('\n')

    # Read in lines one by one and strip away extra spaces, 
    # leading spaces, and trailing spaces and inserting each
    # cleaned up line into a hash table.
    listHash = {}
    re1 = re.compile(' +')
    re2 = re.compile('^ +| +$')
    for stringIn in wordsList:
        term = re2.sub("",re1.sub(" ",stringIn.strip('\n')))
        if term != '':
            listHash[term] = 1
    return listHash

workfolder = os.environ.get('AZUREML_NATIVE_SHARE_DIRECTORY')

# paths to trainQ, testQ and function words.
trainQ_path = os.path.join(workfolder, 'trainQ_part1')
testQ_path = os.path.join(workfolder, 'testQ_part1')
function_words_url = 'https://bostondata.blob.core.windows.net/stackoverflow/function_words.txt'

# load the training and test data.
trainQ = pd.read_csv(trainQ_path, sep='\t', index_col='Id', encoding='latin1')
testQ = pd.read_csv(testQ_path, sep='\t', index_col='Id', encoding='latin1')

# Load the list of non-content bearing function words.
functionwordHash = LoadListAsHash(function_words_url)

CleanedTrainQ = CleanAndSplitText(trainQ)
CleanedTestQ = CleanAndSplitText(testQ)

CleanedTrainQ.head(5)

# Initialize an empty list of learned phrases
# If you have completed a partial run of phrase learning
# and want to add more phrases, you can use the pre-learned 
# phrases as a starting point instead and the new phrases
# will be appended to the list
learnedPhrasesQ = []

# Create a copy of the original text data that will be used during learning
# The copy is needed because the algorithm does in-place replacement of learned
# phrases directly on the text data structure it is provided
phraseTextDataQ = []
for textLine in CleanedTrainQ['LowercaseText']:
    phraseTextDataQ.append(' ' + textLine + ' ')

# Run the phrase learning algorithm.
ApplyPhraseLearning(phraseTextDataQ, learnedPhrasesQ, maxNumPhrases=200, maxPhraseLength=7, maxPhrasesPerIter=50,
                    minCount=5, functionwordHash=functionwordHash)

# Add text with learned phrases back into data frame
CleanedTrainQ['TextWithPhrases'] = phraseTextDataQ

# Apply the phrase learning to test data.
CleanedTestQ['TextWithPhrases'] = ApplyPhraseRewritesInPlace(CleanedTestQ, 'LowercaseText', learnedPhrasesQ)

print("\nHere are some phrases we learned in this part of the tutorial: \n")
print(learnedPhrasesQ[:20])

# reconstitue the text from seperated chunks.
trainQ['TextWithPhrases'] = ReconstituteDocsFromChunks(CleanedTrainQ, 'DocID', 'TextWithPhrases')
testQ['TextWithPhrases'] = ReconstituteDocsFromChunks(CleanedTestQ, 'DocID', 'TextWithPhrases')

def TokenizeText(textData, vocabHash):
    tokenizedText = ''
    for token in textData.split():
        if token in vocabHash:
            tokenizedText += (token.strip() + ',')
    return tokenizedText.strip(',')

# create the vocabulary.
vocabHashQ = CreateVocabForTopicModeling(trainQ['TextWithPhrases'], functionwordHash)

# tokenize the text.
trainQ['Tokens'] = trainQ['TextWithPhrases'].apply(lambda x: TokenizeText(x, vocabHashQ))
testQ['Tokens'] = testQ['TextWithPhrases'].apply(lambda x: TokenizeText(x, vocabHashQ))

trainQ[['AnswerId', 'Tokens']].head(5)

trainQ.to_csv(os.path.join(workfolder, 'trainQ_part2'), sep='\t', header=True, index=True, index_label='Id')
testQ.to_csv(os.path.join(workfolder, 'testQ_part2'), sep='\t', header=True, index=True, index_label='Id')

