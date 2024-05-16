from gensim.models import LsiModel
import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle
import gensim

from pprint import pprint

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LsiModel

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
import itertools as it

import en_core_web_sm

import spacy
nlp = spacy.load('en')

import nltk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer


stopwords = stopwords.words('english')
# lemmatizer = nltk.WordNetLemmatizer()

# # load the finished bag-of-words corpus from disk
# trigram_bow_corpus = MmCorpus('../data/models_data_lower/spacy_trigram_bow_corpus_all.mm') # No POS preprocessing

# # load the finished dictionary from disk
# trigram_dictionary = Dictionary.load('../data/models_data_lower/spacy_trigram_dict_all.dict') # No POS preprocessing

# load the finished bag-of-words corpus from disk
trigram_bow_corpus_POS = MmCorpus('../data/spacy_trigram_bow_corpus_all_POS.mm') # With POS preprocessing

# load the finished dictionary from disk
trigram_dictionary_POS = Dictionary.load('../data/spacy_trigram_dict_all_POS.dict') # With POS preprocessing

get_ipython().run_cell_magic('time', '', "# Create LDA model\n\nwith warnings.catch_warnings():\n    warnings.simplefilter('ignore')\n\n    # workers => sets the parallelism, and should be\n    # set to your number of physical cores minus one\n    lda_alpha_auto = LdaModel(trigram_bow_corpus_POS, \n                                 id2word=trigram_dictionary_POS, \n                                 num_topics=25,\n                             alpha='auto', eta='auto')\n    \n    lda_alpha_auto.save('../data/models_data_lower/spacy_lda_model_POS_alpha_eta_auto')")

# load the finished LDA model from disk
lda = LdaModel.load('../data/models_data_lower/spacy_lda_model_POS_alpha_eta_auto')

topic_names = {1: u'(?)Large Tech Corps (NVIDIA, Splunk, Twitch)',
               2: u'Technical Federal Contracting and Cybersecurity',
               3: u'Financial Risk and Cybersecurity',
               4: u'Web Development (More Frontend)',
               5: u'Social Media Marketing',
               6: u'Fintech, Accounting, and Investing Analysis/Data',
               7: u'(?)Students, Interns, CMS/Marketing, Benefits',
               8: u'Health Care (Data Systems)',
               9: u'Database Administrator',
               10: u'Marketing and Growth Strategy',
               11: u'Quality Assurance and Testing',
               12: u'Data Science',
               13: u'Big Data Engineering',
               14: u'Sales',
               15: u'(?)Large Tech Corps Chaff: Fiserv, Adove, SAP',
               16: u'Flight and Space (Hardware & Software)',
               17: u'Networks, Hardware, Linux',
               18: u'Supervisor, QA, and Process Improvement',
               19: u'Defense Contracting',
               20: u'Social Media Advertising Management',
               21: u'UX and Design',
               22: u'(?)Amazon Engineering/Computing/Robotics/AI',
               23: u'Mobile Developer',
               24: u'DevOps',
               25: u'Payments, Finance, and Blockchain'}

## Visualize

LDAvis_data_filepath = '../models/ldavis_prepared'

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.display(LDAvis_prepared)

# Original functions from processing step, in modeling.ipynb
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

from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
# Load up the bigram and trigram models we trained earlier
bigram_model = Phrases.load('../models/spacy_bigram_model_all_PARSED_POS')
trigram_model = Phrases.load('../models/spacy_trigram_model_all_PARSED_POS')

trigram_dictionary = trigram_dictionary_POS
def vectorize_input(input_doc, bigram_model, trigram_model, trigram_dictionary):
    """
    (1) parse input doc with spaCy, apply text pre-proccessing steps, 
    (3) create a bag-of-words representation (4) create an LDA representation
    """
    
    # parse the review text with spaCy
    parsed_doc = nlp(input_doc)
    
    # lemmatize the text and remove punctuation and whitespace
    unigram_doc = [token.lemma_ for token in parsed_doc
                      if not punct_space(token)]
    
    # apply the first-order and secord-order phrase models
    bigram_doc = bigram_model[unigram_doc]
    trigram_doc = trigram_model[bigram_doc]
    
    # remove any remaining stopwords
    trigram_review = [term for term in trigram_doc
                      if not term in stopwords]
    
    # create a bag-of-words representation
    doc_bow = trigram_dictionary.doc2bow(trigram_doc)

    # create an LDA representation
    document_lda = lda[doc_bow]
    return trigram_review, document_lda

def lda_top_topics(document_lda, topic_names, min_topic_freq=0.05):
    '''
    Print a sorted list of the top topics for a given LDA representation
    '''
    # sort with the most highly related topics first
    sorted_doc_lda = sorted(document_lda, key=lambda review_lda: -review_lda[1])
    
    for topic_number, freq in sorted_doc_lda:
        if freq < min_topic_freq:
            break
            
        # print the most highly related topic names and frequencies
        print('*'*56) 
        print('{:50} {:.3f}'.format(topic_names[topic_number+1],
                                round(freq, 3)))
        print('*'*56)
        for term, term_freq in lda.show_topic(topic_number, topn=10):
            print(u'{:20} {:.3f}'.format(term, round(term_freq, 3)))
        print('\n\n')

def top_match_items(document_lda, topic_names, num_terms=100):
    '''
    Print a sorted list of the top topics for a given LDA representation
    '''
    # sort with the most highly related topics first
    sorted_doc_lda = sorted(document_lda, key=lambda review_lda: -review_lda[1])

    topic_number, freq = sorted_doc_lda[0][0], sorted_doc_lda[0][1]
    print('*'*56)
    print('{:50} {:.3f}'.format(topic_names[topic_number+1],
                            round(freq, 3)))
    print('*'*56)
    for term, term_freq in lda.show_topic(topic_number, topn=num_terms):
        print(u'{:20} {:.3f}'.format(term, round(term_freq, 3)))


def top_match_list(document_lda, topic_names, num_terms=500):
    # Take the above results and just save to a list of the top 500 terms in the topic
    sorted_doc_lda = sorted(document_lda, key=lambda review_lda: -review_lda[1])
    topic_number, freq = sorted_doc_lda[0][0], sorted_doc_lda[0][1]
    print('Highest probability topic:', topic_names[topic_number+1],'\t', round(freq, 3))
    top_topic_skills = []
    for term, term_freq in lda.show_topic(topic_number, topn=num_terms):
        top_topic_skills.append(term)
    return top_topic_skills

def common_skills(top_topic_skills, user_skills):
    return [item for item in top_topic_skills if item in user_skills]

def non_common_skills(top_topic_skills, user_skills):
    return [item for item in top_topic_skills if item not in user_skills]

with open('../data/sample_resume.txt', 'r') as infile:
    sample1 = infile.read()
with open('../data/sample_ds_resume2.txt', 'r') as infile:
    sample2 = infile.read()

def generate_common_skills(input_sample):
    user_skills, my_lda = vectorize_input(input_sample, bigram_model, trigram_model, trigram_dictionary)
    # top_items = top_match_items(my_lda, topic_names)
    # print(top_items)
    skills_list = top_match_list(my_lda, topic_names, num_terms=500)
    print("Top 40 skills user has in common with topic:")
    pprint(common_skills(skills_list, user_skills)[:100])
    print("\n\nTop 40 skills user DOES NOT have in common with topic:")
    pprint(non_common_skills(skills_list, user_skills)[:100])

user_skills, my_lda = vectorize_input(sample1, bigram_model, trigram_model, trigram_dictionary)
print(user_skills)

generate_common_skills(sample2)

