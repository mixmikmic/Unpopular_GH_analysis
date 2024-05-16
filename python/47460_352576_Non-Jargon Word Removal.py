import spacy
import sys
import pprint as pp
import re

from spacy.en import English
from spacy.tokenizer import Tokenizer

nlp = spacy.load('en')

def strip_non_words(tokenized_text):
    return [token for token in tokenized_text if token.is_alpha==True]

# Takes in a bag of words and spits out that same bag of words but without the proper nouns
def strip_proper_nouns(tokenized_text):
    return [token for token in tokenized_text if token.tag_ != 'NNP' and token.tag_ != 'NNPS']

# Takes in a bag of words and removes any of them that are in the top n most common words
def strip_most_common_words(tokenized_text, n_most_common=10000):
    # Build the list of most common words
    most_common_words = []
    google_most_common_words_path = sys.path[1] + '/../Texts/google-10000-english-usa.txt'
    with open(google_most_common_words_path, 'r') as f:
        for i in range(n_most_common):
            most_common_words.append(f.readline().strip())
    # Remove anything in the n most common words
    return [token for token in tokenized_text if token.text.lower() not in most_common_words]

def strip_non_jargon_words(tokenized_text):
    text_no_proper_nouns = strip_proper_nouns(tokenized_text)
    text_no_non_words = strip_non_words(text_no_proper_nouns)
    text_no_common_words = strip_most_common_words(text_no_non_words)
    return text_no_common_words


def load_doc(filepath):
    # Open and read the file
    with open(file_path) as f:
        text = f.read()
    doc = nlp(text)
    return doc

# Here's an example with Gladwell. Shout out to Gladwell!

file_path = sys.path[1] + '/../Rule3/gladwell_latebloomers.txt'
gladwell_doc = load_doc(file_path)

pp.pprint(strip_non_jargon_words(gladwell_doc))

