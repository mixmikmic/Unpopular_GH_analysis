import os
data_switch=1
if(data_switch==0):
    train_dir = '../data/ling-spam/train-mails/'
    email_path = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
else:
    train_dir = '../data/lingspam_public/bare/'
    email_path = []
    email_label = []
    for d in os.listdir(train_dir):
        folder = os.path.join(train_dir,d)
        email_path += [os.path.join(folder,f) for f in os.listdir(folder)]
        email_label += [f[0:3]=='spm' for f in os.listdir(folder)]
print("number of emails",len(email_path))
email_nb = 0 # try 8 for a spam example
print("email file:", email_path[email_nb])
print("email is a spam:", email_label[email_nb])
print(open(email_path[email_nb]).read())

from sklearn.feature_extraction.text import CountVectorizer
countvect = CountVectorizer(input='filename', stop_words='english', lowercase=True)
word_count = countvect.fit_transform(email_path)

print("Number of documents:", len(email_path))
words = countvect.get_feature_names()
print("Number of words:", len(words))
print("Document - words matrix:", word_count.shape)
print("First words:", words[0:100])

from nltk import wordpunct_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
from string import punctuation
class LemmaTokenizer(object):
    def __init__(self, remove_non_words=True):
        self.wnl = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.words = set(words.words())
        self.remove_non_words = remove_non_words
    def __call__(self, doc):
        # tokenize words and punctuation
        word_list = wordpunct_tokenize(doc)
        # remove stopwords
        word_list = [word for word in word_list if word not in self.stopwords]
        # remove non words
        if(self.remove_non_words):
            word_list = [word for word in word_list if word in self.words]
        # remove 1-character words
        word_list = [word for word in word_list if len(word)>1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.wnl.lemmatize(t) for t in word_list]

countvect = CountVectorizer(input='filename',tokenizer=LemmaTokenizer(remove_non_words=True))
word_count = countvect.fit_transform(email_path)
feat2word = {v: k for k, v in countvect.vocabulary_.items()}

print("Number of documents:", len(email_path))
words = countvect.get_feature_names()
print("Number of words:", len(words))
print("Document - words matrix:", word_count.shape)
print("First words:", words[0:100])

mail_number = 0
text = open(email_path[mail_number]).read()
print("Original email:")
print(text)
#print(LemmaTokenizer()(text))
#print(len(set(LemmaTokenizer()(text))))
#print(len([feat2word[i] for i in word_count2[mail_number, :].nonzero()[1]]))
#print(len([word_count2[mail_number, i] for i in word_count2[mail_number, :].nonzero()[1]]))
#print(set([feat2word[i] for i in word_count2[mail_number, :].nonzero()[1]])-set(LemmaTokenizer()(text)))
emailBagOfWords = {feat2word[i]: word_count[mail_number, i] for i in word_count[mail_number, :].nonzero()[1]}
print("Bag of words representation (", len(emailBagOfWords), " words in dict):", sep='')
print(emailBagOfWords)
print("\nVector reprensentation (", word_count[mail_number, :].nonzero()[1].shape[0], " non-zero elements):", sep='')
print(word_count[mail_number, :])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer().fit_transform(word_count)
tfidf.shape

print("email 0:")
print(tfidf[0,:])

import load_spam
spam_data = load_spam.spam_data_loader()
spam_data.load_data()

spam_data.print_email(8)



