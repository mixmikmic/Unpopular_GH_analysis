import os
import urllib.request
import nltk
# The first time you run NLTK you will need to download the 'punkt' models 
# for breaking text strings into individual sentences
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk import tokenize

from azureml.logging import get_azureml_logger
aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
aml_logger.log('amlrealworld.document-collection-analysis.notebook1', 'true')

import pandas 
import re

def download_file_from_blob(filename):
    shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
    save_path = os.path.join(shared_path, filename)

    # Base URL for anonymous read access to Blob Storage container
    STORAGE_CONTAINER = 'https://bostondata.blob.core.windows.net/scenario-document-collection-analysis/'
    url = STORAGE_CONTAINER + filename
    urllib.request.urlretrieve(url, save_path)
    
    
def getData():
    shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']

    data_file = os.path.join(shared_path, DATASET_FILE)
    blacklist_file = os.path.join(shared_path, BLACK_LIST_FILE)
    function_words_file = os.path.join(shared_path, FUNCTION_WORDS_FILE)

    if not os.path.exists(data_file):
        download_file_from_blob(DATASET_FILE)
    if not os.path.exists(blacklist_file):
        download_file_from_blob(BLACK_LIST_FILE)
    if not os.path.exists(function_words_file):
        download_file_from_blob(FUNCTION_WORDS_FILE)

    df = pandas.read_csv(data_file, sep='\t')
    return df

# The dataset file name
# DATASET_FILE = 'small_data.tsv'
DATASET_FILE = 'CongressionalDataAll_Jun_2017.tsv'

# The black list of words to ignore
BLACK_LIST_FILE = 'black_list.txt'

# The non-content bearing function words
FUNCTION_WORDS_FILE = 'function_words.txt'

frame = getData()

print("Total documents in corpus: %d\n" % len(frame))

# Show the first five rows of the data in the frame
frame[0:5]

print(frame['Text'][0])
print('---')
print(frame['Text'][1])
print('---')
print(frame['Text'][2])

def CleanAndSplitText(textDataFrame):

    textDataOut = [] 
   
    # This regular expression is for section headers in the bill summaries that we wish to ignore
    reHeaders = re.compile(r" *TABLE OF CONTENTS:? *"
                           "| *Title [IVXLC]+:? *"
                           "| *Subtitle [A-Z]+:? *"
                           "| *\(Sec\. \d+\) *")

    # This regular expression is for punctuation that we wish to clean out
    # We also will split sentences into smaller phrase like units using this expression
    rePhraseBreaks = re.compile("[\"\!\?\)\]\}\,\:\;\*\-]*\s+\([0-9]+\)\s+[\(\[\{\"\*\-]*"                             
                                "|[\"\!\?\)\]\}\,\:\;\*\-]+\s+[\(\[\{\"\*\-]*"
                                "|\.\.+"
                                "|\s*\-\-+\s*"
                                "|\s+\-\s+"
                                "|\:\:+"
                                "|\s+[\/\(\[\{\"\-\*]+\s*"
                                "|[\,!\?\"\)\(\]\[\}\{\:\;\*](?=[a-zA-Z])"
                                "|[\"\!\?\)\]\}\,\:\;]+[\.]*$"
                             )
    
    # Regex for underbars
    regexUnderbar = re.compile('_')
    
    # Regex for space
    regexSpace = re.compile(' +')
 
    # Regex for sentence final period
    regexPeriod = re.compile("\.$")

    # Iterate through each document and do:
    #    (1) Split documents into sections based on section headers and remove section headers
    #    (2) Split the sections into sentences using NLTK sentence tokenizer
    #    (3) Further split sentences into phrasal units based on punctuation and remove punctuation
    #    (4) Remove sentence final periods when not part of an abbreviation 

    for i in range(0, len(frame)):     
        # Extract one document from frame
        docID = frame['ID'][i]
        docText = str(frame['Text'][i])

        # Set counter for output line count for this document
        lineIndex=0;

        # Split document into sections by finding sections headers and splitting on them 
        sections = reHeaders.split(docText)
        
        for section in sections:
            # Split section into sentence using NLTK tokenizer 
            sentences = tokenize.sent_tokenize(section)
            
            for sentence in sentences:
                # Split each sentence into phrase level chunks based on punctuation
                textSegs = rePhraseBreaks.split(sentence)
                numSegs = len(textSegs)
                
                for j in range(0,numSegs):
                    if len(textSegs[j])>0:
                        # Convert underbars to spaces 
                        # Underbars are reserved for building the compound word phrases                   
                        textSegs[j] = regexUnderbar.sub(" ",textSegs[j])
                    
                        # Split out the words so we can specially handle the last word
                        words = regexSpace.split(textSegs[j])
                        phraseOut = ""
                        # If the last word ends in a period then remove the period
                        words[-1] = regexPeriod.sub("", words[-1])
                        # If the last word is an abbreviation like "U.S."
                        # then add the word final period back on
                        if "\." in words[-1]:
                            words[-1] += "."
                        phraseOut = " ".join(words)  

                        textDataOut.append([docID, lineIndex, phraseOut])
                        lineIndex += 1
                        
    # Convert to pandas frame 
    frameOut = pandas.DataFrame(textDataOut, columns=['DocID', 'DocLine', 'CleanedText'])                      
    
    return frameOut

# Set this to true to run the function
writeFile = True

if writeFile:
    cleanedDataFrame = CleanAndSplitText(frame)

cleanedDataFile = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'], 'CongressionalDocsCleaned.tsv')

if writeFile:
    # Write frame with preprocessed text out to TSV file 
    cleanedDataFrame.to_csv(cleanedDataFile, sep='\t', index=False)
else:
    # Read a cleaned data frame in from a TSV file
    cleanedDataFrame = pandas.read_csv(cleanedDataFile, sep='\t', encoding="ISO-8859-1")

cleanedDataFrame[0:25]

print(cleanedDataFrame['CleanedText'][0])
print(cleanedDataFrame['CleanedText'][1])
print(cleanedDataFrame['CleanedText'][2])

