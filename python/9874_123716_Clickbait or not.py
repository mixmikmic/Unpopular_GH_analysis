import numpy
import sys
import nltk
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


buzzfeed_df = pd.read_json('data/buzzfeed.json')
clickhole_df = pd.read_json('data/clickhole.json')
dose_df =  pd.read_json('data/dose.json')

nytimes_df =  pd.read_json('data/nytimes.json')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
def prepross(header):
    header = (" ").join([word.lower() for word in header.split(" ")])
    header = (" ").join([word for word in header.split(" ") if word not in stopwords.words('english')])
    header = (" ").join([lemmatizer.lemmatize(word) for word in header.split(" ")])
    header = (" ").join(tokenizer.tokenize(header))
    header = (" ").join('NUM' if word in numpy.arange(100) else word for word in header.split(" ") )
    header = (" ").join(word if len(word) > 2 else "" for word in header.split(" ") )
    
    return header.lower()

buzzfeed_df['article_title'] = buzzfeed_df['article_title'].apply(prepross)
clickhole_df['article_title'] = clickhole_df['article_title'].apply(prepross)
dose_df['article_title'] = dose_df['article_title'].apply(prepross)
nytimes_df['article_title'] = nytimes_df['article_title'].apply(prepross)

temp1 = buzzfeed_df[['article_title','clickbait']]
temp2 = clickhole_df[['article_title','clickbait']]
temp3 = dose_df[['article_title','clickbait']]

temp4 = nytimes_df[['article_title','clickbait']]



concat_df = pd.concat([temp1,temp2,temp3,temp4], ignore_index= True)

from sklearn.cross_validation import train_test_split
train, test = train_test_split(concat_df,test_size = 0.3, random_state = 42)

#TFIDF
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = TfidfVectorizer(ngram_range=(1, 3),                          
                             strip_accents='unicode',
                             min_df=2,
                             norm='l2')

#train

X_train = numpy.array(train['article_title'])
Y_train = numpy.array(train['clickbait'])
X_test = numpy.array(test['article_title'])
Y_test = numpy.array(test['clickbait'])

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf = clf.fit(X_train, Y_train)
Y_predicted = clf.predict(X_test)
print metrics.classification_report(Y_test, Y_predicted)







#enter your own title
title = raw_input()
input_df = pd.DataFrame([title],columns=['article_title'])

input_test = vectorizer.transform(numpy.array(input_df['article_title'].apply(prepross)))
if (nb_classifier.predict(input_test)[0] ==  1):
    print "Clickbait with " + str(clf.predict_proba(input_test)[0][1] * 100) + " % probability"
else: 
    print "Article with " + str(clf.predict_proba(input_test)[0][0] * 100) + " % probability"










