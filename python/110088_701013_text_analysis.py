#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object

model_name = "model6"

posts = load_object('objects/',model_name+"-posts")
df    = load_object('objects/',model_name+"-df")

num_posts = len(df['cleantext'])

#get the number of users (minus [deleted])
user_list= df["author"].tolist()
user_dict = {}
for user in user_list:
    if user in user_dict.keys() and user != "[deleted]":
        user_dict[user] =1+user_dict[user]
    else:
        user_dict[user] =1
num_users = len(list(user_dict.keys()))

num_posts

num_users

plain_words = list(df['cleantext'].apply(lambda x: x.split()))

total_phrases =0
for post in posts:
    for phrase in post:
        total_phrases +=1

total_words =0
for post in plain_words:
    for word in post:
        total_words +=1

phrase_dict = {}
for post in posts:
    for phrase in post:
        if phrase in phrase_dict.keys():
            phrase_dict[phrase] =1+phrase_dict[phrase]
        else:
            phrase_dict[phrase] =1

word_dict= {}
for post in plain_words:
    for word in post:
        if word in word_dict.keys():
            word_dict[word] =1+word_dict[word]
        else:
            word_dict[word] =1

# Total words in the corpus
total_words

# Total phrases in the corpus
total_phrases

# Total vocabulary of words
len(list(word_dict))

# Total vocabulary of phrases
len(list(phrase_dict))

phrases = list(phrase_dict.keys())
phrase_freq_count            = 0
filtered_phrase_freq_count   = 0
phrase_unique_count          = 0
filtered_phrase_unique_count = 0
for phrase in phrases:
    count = phrase_dict[phrase]
    phrase_freq_count            += count
    filtered_phrase_freq_count   += count if count >= 10 else 0
    phrase_unique_count          += 1
    filtered_phrase_unique_count += 1 if count >= 10 else 0

words = list(word_dict.keys())
word_freq_count            = 0
filtered_word_freq_count   = 0
word_unique_count          = 0
filtered_word_unique_count = 0
for word in words:
    count = word_dict[word]
    word_freq_count            += count
    filtered_word_freq_count   += count if count >= 10 else 0
    word_unique_count          += 1
    filtered_word_unique_count += 1 if count >= 10 else 0

# Total number of tokens, including phrases and words
phrase_freq_count

# Total number of words, not including phrases
word_freq_count

# Number of words removed by including them in phrases
word_freq_count-phrase_freq_count

# Total number of tokens after filtering, including phrases and words
filtered_phrase_freq_count

# Total number of tokens after filtering, including just words
filtered_word_freq_count

# Check that unique count was calculated correctlly
phrase_unique_count == len(phrase_dict) and word_unique_count == len(word_dict)

# the size of the vocabulary after filtering phrases
filtered_phrase_unique_count

# the number of unique tokens removed by filtering
phrase_unique_count - filtered_phrase_unique_count

# The percent of total tokens removed
str((phrase_freq_count-filtered_phrase_freq_count)/phrase_freq_count*100) + str("%")

# The percent of total tokens preserved
str(100 -100*(phrase_freq_count-filtered_phrase_freq_count)/phrase_freq_count) + str("%")

model = gensim.models.Word2Vec.load('models/'+model_name+'.model')

vocab_list = sorted(list(model.wv.vocab))

# Ensure model has correct number of unique words
len(vocab_list)==filtered_phrase_unique_count

model_freq_count = 0
for word in vocab_list:
    model_freq_count += model.wv.vocab[word].count

# Ensure that the total count of the model's words is the total count of the filtered words
model_freq_count==filtered_phrase_freq_count

