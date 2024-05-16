get_ipython().magic('run helper_functions.py')
get_ipython().magic('run tweepy_wrapper.py')
get_ipython().magic('run s3.py')
get_ipython().magic('run mongo.py')
get_ipython().magic('run df_functions.py')

import pandas as pd
import string
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words("english")+["rt", "via","-»","--»","--","---","-->","<--","->","<-","«--","«","«-","»","«»"]

gabr_tweets = extract_users_tweets("gabr_ibrahim", 2000)

gabr_dict = dict()
gabr_dict['gabr_ibrahim'] = {"content" : [], "hashtags" : [], "retweet_count": [], "favorite_count": []}

for tweet in gabr_tweets:
    text = extract_text(tweet)
    hashtags = extract_hashtags(tweet)
    rts = tweet.retweet_count
    fav = tweet.favorite_count
    
    gabr_dict['gabr_ibrahim']['content'].append(text)
    gabr_dict['gabr_ibrahim']['hashtags'].extend(hashtags)
    gabr_dict['gabr_ibrahim']["retweet_count"].append(rts)
    gabr_dict['gabr_ibrahim']["favorite_count"].append(fav)

gabr_tweets_df = pd.DataFrame.from_dict(gabr_dict, orient='index')

gabr_tweets_df.head()

clean_gabr_tweets = filtration(gabr_tweets_df, "content")

clean_gabr_tweets = dataframe_to_dict(clean_gabr_tweets)

clean_gabr_tweets #this is a list of 1 dictionary

import spacy
import nltk
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim
from collections import Counter
from gensim.corpora.dictionary import Dictionary
nlp = spacy.load('en')

gabr_tweets = clean_gabr_tweets[0]['gabr_ibrahim']['content']

gabr_tweets[:5]

tokenized_tweets = []
for tweet in gabr_tweets:
    tokenized_tweet = nlp(tweet)
    
    tweet = "" # we want to keep each tweet seperate
    
    for token in tokenized_tweet:
        if token.is_space:
            continue
        elif token.is_punct:
            continue
        elif token.is_stop:
            continue
        elif token.is_digit:
            continue
        elif len(token) == 1:
            continue
        elif len(token) == 2:
            continue
        else:
            tweet += str(token.lemma_) + " " #creating lemmatized version of tweet
        
    tokenized_tweets.append(tweet)
tokenized_tweets = list(map(str.strip, tokenized_tweets)) # strip whitespace
tokenized_tweets = [x for x in tokenized_tweets if x != ""] # remove empty entries

tokenized_tweets[:5] # you can see how this is different to the raw tweets!

clean_gabr_tweets[0]['gabr_ibrahim']['tokenized_tweets'] = tokenized_tweets

clean_gabr_tweets_df = pd.DataFrame.from_dict(clean_gabr_tweets[0], orient='index')

clean_gabr_tweets_df.head()

clean_gabr_tweets_df = filtration(clean_gabr_tweets_df, "tokenized_tweets")

clean_gabr_tweets = dataframe_to_dict(clean_gabr_tweets_df)

clean_gabr_tweets[0]['gabr_ibrahim']['tokenized_tweets'][:5]

list_of_tweets_gabr = clean_gabr_tweets[0]['gabr_ibrahim']['tokenized_tweets']

gensim_format_tweets = []
for tweet in list_of_tweets_gabr:
    list_form = tweet.split()
    gensim_format_tweets.append(list_form)

gensim_format_tweets[:5]

gensim_dictionary = Dictionary(gensim_format_tweets)

gensim_dictionary.filter_extremes(no_below=10, no_above=0.4)
gensim_dictionary.compactify() # remove gaps after words that were removed

get_ipython().system('pwd')

file_path_corpus = "/home/igabr/new-project-4"

def bag_of_words_generator(lst, dictionary):
    assert type(dictionary) == Dictionary, "Please enter a Gensim Dictionary"
    for i in lst: 
        yield dictionary.doc2bow(i)

MmCorpus.serialize(file_path_corpus+"{}.mm".format("gabr_ibrahim"), bag_of_words_generator(gensim_format_tweets, gensim_dictionary))

corpus = MmCorpus(file_path_corpus+"{}.mm".format("gabr_ibrahim"))

corpus.num_terms # the number of terms in our corpus!

corpus.num_docs # the number of documets. These are the number of tweets!

lda = LdaMulticore(corpus, num_topics=30, id2word=gensim_dictionary, chunksize=2000, workers=100, passes=100)

lda.save(file_path_corpus+"lda_model_{}".format("gabr_ibrahim"))

lda = LdaMulticore.load(file_path_corpus+"lda_model_{}".format("gabr_ibrahim"))

from collections import Counter

word_list = []

for i in range(30):
    for term, frequency in lda.show_topic(i, topn=100): #returns top 100 words for a topic
        if frequency != 0:
            word_list.append(term)
temp = Counter(word_list)

len(temp)

# This can be done later to help filter the important words.
important_words = []
for k, v in temp.items():
    if v >= 10:
        if k not in nltk_stopwords:
            doc = nlp(k)
            
            for token in doc:
                if not token.is_stop:
                    if len(token) != 2:
                        important_words.append(k)

important_words

len(important_words)

clean_gabr_tweets[0]['gabr_ibrahim'].keys()

clean_gabr_tweets[0]['gabr_ibrahim']['LDA'] = temp

pickle_object(clean_gabr_tweets, "gabr_ibrahim_tweets_LDA_Complete")



