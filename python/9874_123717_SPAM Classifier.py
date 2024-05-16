import nltk
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

messages = [line.rstrip() for line in open('SMSSpamCollection')]

print len(messages)

for num,messages in enumerate (messages[0:10]):
    print num, messages
    print '\n'

import pandas
messages = pandas.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()

messages.describe()

messages.groupby('label').describe()

messages['length'] = messages['message'].apply(len)
messages.head()

messages['length'].plot(bins=100, kind='hist') 

messages.length.describe()

messages[messages['length'] == 910]['message'].iloc[0]

messages.hist(column='length', by='label', bins=50,figsize=(10,4))

from nltk.corpus import stopwords
import string

def text_process(mess):
    """

    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages['message'].head(5).apply(text_process)

messages.head()

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print len(bow_transformer.vocabulary_)

message4 = messages['message'][3]
print message4

bow4 = bow_transformer.transform([message4])
print bow4
print bow4.shape

print bow_transformer.get_feature_names()[4073]
print bow_transformer.get_feature_names()[9570]

messages_bow = bow_transformer.transform(messages['message'])
print 'Shape of Sparse Matrix: ', messages_bow.shape
print 'Amount of Non-Zero occurences: ', messages_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4

print tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]
print tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

messages_tfidf = tfidf_transformer.transform(messages_bow)
print messages_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

print 'predicted:', spam_detect_model.predict(tfidf4)[0]
print 'expected:', messages.label[3]

all_predictions = spam_detect_model.predict(messages_tfidf)
print all_predictions

from sklearn.metrics import classification_report
print classification_report(messages['label'], all_predictions)

from sklearn.cross_validation import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print classification_report(predictions,label_test)



