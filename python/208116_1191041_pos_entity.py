from sklearn.datasets.base import Bunch
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

import re
from stop_words import get_stop_words

import numpy as np
import pandas as pd

import time
import numpy as np
import matplotlib.pyplot as plt

# Some NLTK specifics
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer


import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim

from pprint import pprint

import codecs
import os
import time

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')

#CONNECTING TO THE DATASET
CORPUS_ROOT = "/Users/goodgame/desktop/Shift/match/reqs/jobs_text_meat_only/"

def load_data(root=CORPUS_ROOT):
    """
    Loads the text data into memory using the bundle dataset structure.
    Note that on larger corpora, memory safe CorpusReaders should be used.
    """

    # Open the README and store
    with open(os.path.join(root, 'README'), 'r') as readme:
        DESCR = readme.read()

    # Iterate through all the categories
    # Read the HTML into the data and store the category in target
    data      = []
    target    = []
    filenames = []

    for category in os.listdir(root):
        if category == "README": continue # Skip the README
        if category == ".DS_Store": continue # Skip the .DS_Store file
        for doc in os.listdir(os.path.join(root, category)):
            if doc == ".DS_Store": continue
            fname = os.path.join(root, category, doc)

            # Store information about document
            filenames.append(fname)
            target.append(category)
            with codecs.open(fname, 'r', 'ISO-8859-1') as f:
                data.append(f.read())
            # Read data and store in data list
            # with open(fname, 'r') as f:
            #     data.append(f.read())

    return Bunch(
        data=data,
        target=target,
        filenames=filenames,
        target_names=frozenset(target),
        DESCR=DESCR,
    )

dataset = load_data()

#print out the readme file
print(dataset.DESCR)
#Remember to create a README file and place it inside your CORPUS ROOT directory if you haven't already done so.

#print the number of records in the dataset
print("The number of instances is ", len(dataset.data), "\n")

job_str = '''Data Scientist/Machine Learning Engineer, Adaptive Authentication
Position Description:

This is an opportunity to join our fast-growing Adaptive Authentication team to develop cutting-edge risk-based adaptive authentication policies. We are looking for a Data Scientist/Machine Learning Engineer to build large-scale distributed systems while using machine learning to solve business problems. The ideal candidate has experience building models from complex systems, developing enterprise-grade software in an object-oriented language, and experience or knowledge in security, authentication or identity.

Our elite team is fast, innovative and flexible; with a weekly release cycle and individual ownership we expect great things from our engineering and reward them with stimulating new projects and emerging technologies.


Job Duties and Responsibilities:

Build and own models that identify risk associated with anomalous activity in the cloud for authentication
Build Machine Learning pipelines for training and deploying models at scale
Analyze activity data in the cloud for new behavioral patterns
Partner with product leaders to define requirements for building models
Work closely with engineering lead and management to scope and plan engineering efforts
Test-driven development, design and code reviews

Required Skills:

2+ years of Data Science/Machine Learning experience
Skilled in using machine learning algorithms for classification and regression
5+ years of software development experience in an object-oriented language building highly-reliable, mission-critical software
Excellent grasp of software engineering principles
Experience with multi-factor authentication, security, or identity is a plus

Education:

B.S, M.S, or Ph.D. in computer science, data science, machine learning, information retrieval, math or equivalent work experience

 

Okta is an Equal Opportunity Employer'''

stopwords = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()

def normalise(word):
    word = word.lower().replace('/','').replace('-','').replace('•','')
    # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
    word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term

# combine functions above
def noun_phrases(text):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')    
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    toks = tokenizer.tokenize(text)
    postoks = nltk.tag.pos_tag(toks)
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    bad_words = ['opportunity', 'ideal candidate', 'team', 'year', 'knowledge','experience']
    clean_terms = []
    
    for term in terms:
        term = ' '.join(term).replace('\n','').replace(',','').replace('(','')
        term = term.replace(')','')
        term = term.strip()
        if term not in bad_words:
            clean_terms.append(term)
    return clean_terms



get_ipython().run_cell_magic('time', '', 'parsed_dataset = []\ntitles = []\n\ndef parse_input_docs(doc):\n    return \' \'.join(noun_phrases(doc))\n\nfor item in dataset.data:\n    titles.append(item.split("\\n",2)[0])\n    copy_minus_title = item.split("\\n",2)[2]\n    parsed_dataset.append(parse_input_docs(copy_minus_title))\nprint(len(parsed_dataset))\nprint(titles[0])')

# Use Gensim Phrases with POS-tagged JDs
# phrases = gensim.models.Phrases(list_pos_jd)
phrases = gensim.models.Phrases(parsed_dataset)
bigram = gensim.models.phrases.Phraser(phrases)

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# porter_stemmer = PorterStemmer()

def lemmatizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    return words

stop_words = text.ENGLISH_STOP_WORDS

# TF-IDF transformation in sklearn

pos_vect = TfidfVectorizer(stop_words=stop_words, tokenizer=lemmatizer, ngram_range=(1,2), analyzer='word')  
pos_tfidf = pos_vect.fit_transform(parsed_dataset)
print("\n Here are the dimensions of our two-gram dataset: \n", pos_tfidf.shape, "\n")

true_k = 100
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(pos_tfidf)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = pos_vect.get_feature_names()
for i in range(true_k):
    print("\n\nCluster %d:" % i, "\n")
    for ind in order_centroids[i, :40]:
        print(' %s' % terms[ind])

print(parsed_dataset[0])

documents = parsed_dataset

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

get_ipython().system('rm /tmp/parsed.mm')

dictionary = gensim.corpora.Dictionary(texts)
dictionary.save('/tmp/parsed.dict')  # store the dictionary, for future reference

# Look at token IDs
# print(dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # All three words appear in the dictionary

corpus = [dictionary.doc2bow(text) for text in texts]
gensim.corpora.MmCorpus.serialize('/tmp/parsed.mm', corpus)  # store to disk, for later use

if (os.path.exists("/tmp/parsed.dict")):
    dictionary = gensim.corpora.Dictionary.load('/tmp/parsed.dict')
    corpus = gensim.corpora.MmCorpus('/tmp/parsed.mm')
    print("Used saved dictionary and corpus")
else:
    print("Please run first tutorial to generate data set")

print(dictionary[0])
print(dictionary[1])
print(dictionary[2])

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)

# hdp = gensim.models.HdpModel(corpus, id2word=dictionary)

print(job_str)

doc = parse_input_docs(job_str)
print(doc, "\n\n")
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)

index = gensim.similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

get_ipython().system('rm /tmp/parsed.index')
index.save('/tmp/parsed.index')

sims = index[vec_lsi]
sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
for item in sims_sorted[:5]:
    print(titles[item[0]],"\n\tIndex:",item[0],"\n\tSimilarity:",item[1])



print(documents[174])

# Word similarities in a high-scored document:
print([word for word in documents[821].split() if word in doc])

# Word similarities in a low-scored document
print([word for word in documents[174].split() if word in doc])

with open('sample_resume.txt', 'r') as infile:
    resume = infile.read()
doc = parse_input_docs(resume)
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space

sims = index[vec_lsi]
sims_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
for item in sims_sorted[:5]:
    print(titles[item[0]],"\n\tIndex:",item[0],"\n\tSimilarity:",item[1])

# Load Google's pre-trained Word2Vec model.
word2vec = gensim.models.KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)  

word2vec.wv['product']  # numpy vector of a word

word2vec.wv.most_similar(positive=['woman', 'king'], negative=['man'])

phrases = gensim.models.Phrases(' '.join(parsed_dataset))

bigram = gensim.models.phrases.Phraser(phrases)

sent = ['machine','learning','is','so','hot','right','now']
print(bigram[sent])

from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

import spacy
import pandas as pd
import itertools as it
import en_core_web_sm

import spacy
nlp = spacy.load('en')

parsed_review = nlp(job_str)
print(parsed_review)

# for num, sentence in enumerate(parsed_review.sents):
#     print('Sentence {}:'.format(num + 1))
#     print(sentence)
#     print()

# token_text = [token.orth_ for token in parsed_review]
# token_pos = [token.pos_ for token in parsed_review]

# pd.DataFrame(list(zip(token_text, token_pos)),
#              columns=['token_text', 'part_of_speech'])

# token_lemma = [token.lemma_ for token in parsed_review]
# token_shape = [token.shape_ for token in parsed_review]

# pd.DataFrame(list(zip(token_text, token_lemma, token_shape)),
#              columns=['token_text', 'token_lemma', 'token_shape'])

# token_entity_type = [token.ent_type_ for token in parsed_review]
# token_entity_iob = [token.ent_iob_ for token in parsed_review]

# pd.DataFrame(list(zip(token_text, token_entity_type, token_entity_iob)),
#              columns=['token_text', 'entity_type', 'inside_outside_begin'])


# token_attributes = [(token.orth_,
#                      token.prob,
#                      token.is_stop,
#                      token.is_punct,
#                      token.is_space,
#                      token.like_num,
#                      token.is_oov)
#                     for token in parsed_review]

# df = pd.DataFrame(token_attributes,
#                   columns=['text',
#                            'log_probability',
#                            'stop?',
#                            'punctuation?',
#                            'whitespace?',
#                            'number?',
#                            'out of vocab.?'])

# df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?']
#                                        .applymap(lambda x: u'Yes' if x else u''))
                                               
# df

from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def line_review(filename):
    """
    SRG: modified for a list
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    for review in filename:
        yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=4):
        
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

get_ipython().run_cell_magic('time', '', "\nimport codecs\n# This is time consuming; make the if statement True to run\nif 0 == 0:\n    with codecs.open('spacy_parsed_jobs_PARSED.txt', 'w', encoding='utf_8') as f:\n        for sentence in parsed_dataset:\n            f.write(sentence + '\\n')")

unigram_sentences = LineSentence('spacy_parsed_jobs_PARSED.txt')

for unigram_sentence in it.islice(unigram_sentences, 230, 240):
    print(u' '.join(unigram_sentence))
    print(u'')

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute modeling yourself.\nif 0 == 0:\n\n    bigram_model = Phrases(unigram_sentences)\n\n    bigram_model.save('spacy_bigram_model_all_PARSED')\n    \n# load the finished model from disk\nbigram_model = Phrases.load('spacy_bigram_model_all_PARSED')")

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 0:\n\n    with codecs.open('spacy_bigram_sentences_PARSED.txt', 'w', encoding='utf_8') as f:\n        \n        for unigram_sentence in unigram_sentences:\n            \n            bigram_sentence = u' '.join(bigram_model[unigram_sentence])\n            \n            f.write(bigram_sentence + '\\n')")

bigram_sentences = LineSentence('spacy_bigram_sentences_PARSED.txt')

for bigram_sentence in it.islice(bigram_sentences, 240, 250):
    print(u' '.join(bigram_sentence))
    print(u'')

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute modeling yourself.\nif 0 == 0:\n\n    trigram_model = Phrases(bigram_sentences)\n\n    trigram_model.save('spacy_trigram_model_all_PARSED')\n    \n# load the finished model from disk\ntrigram_model = Phrases.load('spacy_trigram_model_all_PARSED')")

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 0:\n\n    with codecs.open('spacy_trigram_sentences_PARSED.txt', 'w', encoding='utf_8') as f:\n        \n        for bigram_sentence in bigram_sentences:\n            \n            trigram_sentence = u' '.join(trigram_model[bigram_sentence])\n            \n            f.write(trigram_sentence + '\\n')")

trigram_sentences = LineSentence('spacy_trigram_sentences_PARSED.txt')

for trigram_sentence in it.islice(trigram_sentences, 240, 250):
    print(u' '.join(trigram_sentence))
    print(u'')

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to execute data prep yourself.\nif 0 == 0:\n\n    with codecs.open('spacy_trigram_transformed_reviews_all_PARSED.txt', 'w', encoding='utf_8') as f:\n        \n        for parsed_review in nlp.pipe(line_review(dataset.data),\n                                      batch_size=10000, n_threads=4):\n            \n            # lemmatize the text, removing punctuation and whitespace\n            unigram_review = [token.lemma_ for token in parsed_review\n                              if not punct_space(token)]\n            \n            # apply the first-order and second-order phrase models\n            bigram_review = bigram_model[unigram_review]\n            trigram_review = trigram_model[bigram_review]\n            \n            # remove any remaining stopwords\n            trigram_review = [term for term in trigram_review\n                              if term not in stopwords]\n            \n            # write the transformed review as a line in the new file\n            trigram_review = u' '.join(trigram_review)\n            f.write(trigram_review + '\\n')")

print(u'Original:' + u'\n')

for review in it.islice(line_review(dataset.data), 11, 12):
    print(review)

print(u'----' + u'\n')
print(u'Transformed:' + u'\n')

with codecs.open('spacy_trigram_transformed_reviews_all.txt', encoding='utf_8') as f:
    for review in it.islice(f, 11, 12):
        print(review)

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LsiModel

import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to learn the dictionary yourself.\nif 0 == 0:\n\n    trigram_reviews = LineSentence('spacy_trigram_sentences_PARSED.txt')\n\n    # learn the dictionary by iterating over all of the reviews\n    trigram_dictionary = Dictionary(trigram_reviews)\n    \n    # filter tokens that are very rare or too common from\n    # the dictionary (filter_extremes) and reassign integer ids (compactify)\n    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)\n    trigram_dictionary.compactify()\n\n    trigram_dictionary.save('spacy_trigram_dict_all.dict')\n    \n# load the finished dictionary from disk\ntrigram_dictionary = Dictionary.load('spacy_trigram_dict_all.dict')")

def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """    
    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to build the bag-of-words corpus yourself.\nif 0 == 0:\n\n    # generate bag-of-words representations for\n    # all reviews and save them as a matrix\n    MmCorpus.serialize('spacy_trigram_bow_corpus_all.mm',\n                       trigram_bow_generator('spacy_trigram_sentences_PARSED.txt'))\n    \n# load the finished bag-of-words corpus from disk\ntrigram_bow_corpus = MmCorpus('spacy_trigram_bow_corpus_all.mm')")

get_ipython().run_cell_magic('time', '', "\n# this is a bit time consuming - make the if statement True\n# if you want to train the LDA model yourself.\nif 0 == 0:\n\n    with warnings.catch_warnings():\n        warnings.simplefilter('ignore')\n        \n        # workers => sets the parallelism, and should be\n        # set to your number of physical cores minus one\n        lsi = gensim.models.LsiModel(trigram_bow_corpus, \n                                     id2word=trigram_dictionary, \n                                     num_topics=500)\n    \n    lsi.save('spacy_lsi_model_all')\n    \n# load the finished LDA model from disk\nlsi = LsiModel.load('spacy_lsi_model_all')")

def explore_topic(topic_number, topn=10):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
        
#     print(u'{:20} {}'.format(u'term', u'frequency') + u'')

    for term, frequency in lsi.show_topic(topic_number, topn=10):
        print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))

for i in range(600):
    print("\n\nTopic %s" % str(i+1))
    explore_topic(topic_number=i)



