import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import *
import scipy

import nlp

data = nlp.proc_text()
data.head()

data['process'][2627]

vectorizer = TfidfVectorizer(analyzer = "word", max_df=1.0, min_df=.03)
#max_df is the fraction of documents that must have a word before it is ignored.
#min_df is the fraction of documents that must have a word for it to be considered.
#norm="l2" normalizes each document vector to a (pythagorean) length of 1.
clean_text = data["process"]
weighted_words = vectorizer.fit_transform(clean_text)

#also get the bag of words without weighting for comparison
vectorizer2 = CountVectorizer(analyzer = "word",min_df=.03)
unweighted_words = vectorizer2.fit_transform(clean_text)

unweighted_words.shape

#Want a function to list words alongside their frequency
def get_word_frequency(sparse_matrix,doc,word_list):
    #find number of distinct words in given document
    num_words = sparse_matrix[doc,:].getnnz()
    #initialize DataFrame
    word_frequency = pd.DataFrame(index=range(num_words), columns=['word','frequency'])
    #convert to another kind of sparse matrix
    cx = scipy.sparse.coo_matrix(sparse_matrix[doc,:])
    #Loop over nonzero elements in the sparse matrix
    #with i = column number, j = weight, and k being the appropriate row of the DataFrame
    for i,j,k in zip(cx.col,cx.data,range(num_words)):
        word_frequency['word'][k] = word_list[i]
        word_frequency['frequency'][k] = j
        
    #Finally, sort the DataFrame
    word_frequency.sort_values('frequency',inplace=True,ascending=False)
    return word_frequency

doc_number = 2
test_freq = get_word_frequency(unweighted_words,doc_number,vectorizer2.get_feature_names())
test_weight = get_word_frequency(weighted_words,doc_number,vectorizer.get_feature_names())
print(str(doc_number) + "th document, before TF-IDF:\n",test_freq[0:10])
print("After TF-IDF:\n",test_weight[0:10])

data.to_pickle("processed_10k_articles.pkl")

numpy.save("document_term_matrix",unweighted_words,allow_pickle=True)

pd.DataFrame(vectorizer.get_feature_names()).to_pickle("term_list.pkl")

processed_10k_articles = pd.read_pickle("processed_10k_articles.pkl")
document_term_matrix = numpy.reshape(numpy.load("document_term_matrix.npy"),(1))[0]
term_list = pd.read_pickle("term_list.pkl")[0].tolist()

