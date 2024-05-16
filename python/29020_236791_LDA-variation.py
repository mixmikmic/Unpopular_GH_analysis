import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import *
import lda #use pip install lda 
from collections import Counter

data = pd.read_pickle("processed_10k_articles.pkl")
titles = [word for word in data.title]

#from Tristan's code:
#putting the code first 
#first generate the bag of words.  This has no TF-IDF weighting yet.
#Only include words that occur in at least 5% of documents.
vectorizer = CountVectorizer(analyzer = "word",min_df=0.05) #0.05
clean_text = [' '.join( (txt.split())[0: min(500, len(txt.split()))])  for txt in data['process'] ]  #data["process"]
unweighted_words = vectorizer.fit_transform(clean_text)
terms_matrix = unweighted_words.toarray()
vocabulary  = vectorizer.vocabulary_ # the words selected 
vocab = [w for w in vocabulary]

model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)
model.fit(terms_matrix)  # model.fit_transform(X) is also available

topic_word = model.topic_word_ 
topic_word.shape # all elements are >= 0

doc_topic = model.doc_topic_
doc_topic.shape # all elements are >= 0

# looking at the topics that are produced
n_top_words = 5
for i, topic_dist in enumerate(topic_word):
    tmp_topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: top words: {} \n (article with highest weight: {})'.format(    i, ' '.join(tmp_topic_words), titles[doc_topic[:,i].argmax()]  ))

# looking at articles
for i in range(10):
#     print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
     print("{} (top topic: {})".format(titles[i], doc_topic[i].argsort()[-2:]))

# top 40 most informative words in the vocabulary
voc_var = [topic_word[:,i].var() for i in range(topic_word[:,:].shape[1]) ]
tophowmany = 40
print('top %i most informative words:'%tophowmany)
top_informative_words = np.asarray(voc_var).argsort()[::-1][:tophowmany]
for i in top_informative_words:
    print("%10s, highest-weight topic:%3d, lowest-weight topic:%d"%(vocab[i], topic_word[:,i].argmax(),  topic_word[:,i].argmin() ))

w = pd.read_html('https://en.wikipedia.org/wiki/' + 'Pierre-Simon_Laplace',flavor='bs4')

v = [word for word in w if w is not np.NaN]

for i in range(len(w)):
     print(w[i].shape)
     

w[6].iloc[0,0]

import wikipedia

print( wikipedia.summary("bilogy",sentences = 1000) )

