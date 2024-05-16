get_ipython().magic('matplotlib inline')

import os 
import requests 

WAPO = "http://wpo.st/"

def fetch_wapo(sid="ciSa2"):
    url = WAPO + sid 
    res = requests.get(url) 
    return res.text

story = fetch_wapo()

print(story)

from bs4 import BeautifulSoup
from readability.readability import Document

def extract(html):
    article = Document(html).summary()
    soup = BeautifulSoup(article, 'lxml')
    
    return soup.get_text()

story = extract(story)

print(story)

import nltk 

def tokenize(text):
    for sent in nltk.sent_tokenize(text):
        yield list(nltk.word_tokenize(sent))

story = list(tokenize(story))

for sent in story: print(sent)

def tag(sents):
    for sent in sents:
        yield list(nltk.pos_tag(sent))

story = list(tag(story))

for sent in story: print(sent)

from nltk.corpus import wordnet as wn

lemmatizer = nltk.WordNetLemmatizer()

def tagwn(tag):
    return {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)


def lemmatize(tagged_sents):
    for sent in tagged_sents:
        for token, tag in sent:
            yield lemmatizer.lemmatize(token, tagwn(tag))


story = list(lemmatize(story))

print(story)

from string import punctuation
from nltk.corpus import stopwords 

punctuation = set(punctuation)
stopwords = set(stopwords.words('english'))

def normalize(tokens):
    for token in tokens:
        token = token.lower()
        if not all(char in punctuation for char in token):
            if token not in stopwords:
                yield token
        

story = list(normalize(story))

print(story)

import string
import pickle 

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

CORPUS_PATH = "data/baleen_sample"
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

class PickledCorpus(CategorizedCorpusReader, CorpusReader):
    
    def __init__(self, root, fileids=PKL_PATTERN, cat_pattern=CAT_PATTERN):
        CategorizedCorpusReader.__init__(self, {"cat_pattern": cat_pattern})
        CorpusReader.__init__(self, root, fileids)
        
        self.punct = set(string.punctuation) | {'“', '—', '’', '”', '…'}
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.wordnet = nltk.WordNetLemmatizer() 
    
    def _resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories=categories)
        
        if fileids is None:
            return self.fileids() 
        
        return fileids
    
    def lemmatize(self, token, tag):
        token = token.lower()
        
        if token not in self.stopwords:
            if not all(c in self.punct for c in token):
                tag =  {
                    'N': wn.NOUN,
                    'V': wn.VERB,
                    'R': wn.ADV,
                    'J': wn.ADJ
                }.get(tag[0], wn.NOUN)
                return self.wordnet.lemmatize(token, tag)
    
    def tokenize(self, doc):
        # Expects a preprocessed document, removes stopwords and punctuation
        # makes all tokens lowercase and lemmatizes them. 
        return list(filter(None, [
            self.lemmatize(token, tag)
            for paragraph in doc 
            for sentence in paragraph 
            for token, tag in sentence 
        ]))
    
    def docs(self, fileids=None, categories=None):
        # Resolve the fileids and the categories
        fileids = self._resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield self.tokenize(pickle.load(f))
    
    def labels(self, fileids=None, categories=None):
        fileids = self._resolve(fileids, categories)
        for fid in fileids:
            yield self.categories(fid)[0]

corpus = PickledCorpus('data/baleen_sample')

print("{} documents in {} categories".format(len(corpus.fileids()), len(corpus.categories())))

from nltk import ConditionalFreqDist

words = ConditionalFreqDist()

for doc, label in zip(corpus.docs(), corpus.labels()):
    for word in doc:
        words[label][word] += 1

for label, counts in words.items():
    print("{}: {:,} vocabulary and {:,} words".format(
        label, len(counts), sum(counts.values())
    ))

from sklearn.manifold import TSNE 
from sklearn.pipeline import Pipeline 
from sklearn.decomposition import TruncatedSVD 
from sklearn.feature_extraction.text import CountVectorizer 

cluster = Pipeline([
        ('vect', CountVectorizer(tokenizer=lambda x: x, preprocessor=None, lowercase=False)), 
        ('svd', TruncatedSVD(n_components=50)), 
        ('tsne', TSNE(n_components=2))
    ])

docs = cluster.fit_transform(list(corpus.docs()))

import seaborn as sns
import matplotlib.pyplot as plt 

from collections import defaultdict 

sns.set_style('whitegrid')
sns.set_context('notebook')

colors = {
    "design": "#e74c3c",
    "tech": "#3498db",
    "business": "#27ae60",
    "gaming": "#f1c40f",
    "politics": "#2c3e50",
    "news": "#bdc3c7",
    "cooking": "#d35400",
    "data_science": "#1abc9c",
    "sports": "#e67e22",
    "cinema": "#8e44ad",
    "books": "#c0392b",
    "do_it_yourself": "#34495e",
}

series = defaultdict(lambda: {'x':[], 'y':[]})
for idx, label in enumerate(corpus.labels()):
    x, y = docs[idx]
    series[label]['x'].append(x)
    series[label]['y'].append(y)

    
fig = plt.figure(figsize=(12,6))
ax = plt.subplot(111)
    
for label, points in series.items():
    ax.scatter(points['x'], points['y'], c=colors[label], alpha=0.7, label=label)

# Add a title 
plt.title("TSNE Projection of the Baleen Corpus")
    
# Remove the ticks 
plt.yticks([])
plt.xticks([])

# Add the legend 
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

hobbies = ['gaming', 'cooking', 'sports', 'cinema', 'books', 'do_it_yourself']

X = list(corpus.docs(categories=hobbies))
y = list(corpus.labels(categories=hobbies))

# Models 
from sklearn.linear_model import SGDClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.ensemble import RandomForestClassifier

# Transformers 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.pipeline import Pipeline 

# Evaluation 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

def identity(words): 
    return words 

# SVM Classifier 
svm = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('svm', SGDClassifier()), 
    ])

yhat = cross_val_predict(svm, X, y, cv=12)
print(classification_report(y, yhat))

# Logistic Regression 
logit = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('logit', LogisticRegression()), 
    ])

yhat = cross_val_predict(logit, X, y, cv=12)
print(classification_report(y, yhat))

# Naive Bayes
nbayes = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('nbayes', MultinomialNB()), 
    ])

yhat = cross_val_predict(nbayes, X, y, cv=12)
print(classification_report(y, yhat))

# Random Forest 
trees = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('trees', RandomForestClassifier()), 
    ])

yhat = cross_val_predict(trees, X, y, cv=12)
print(classification_report(y, yhat))

def build_model(path, corpus):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('svm', SGDClassifier(loss='log')), 
    ])
    
    # Train model on the entire data set 
    X = list(corpus.docs(categories=hobbies))
    y = list(corpus.labels(categories=hobbies))
    model.fit(X, y)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)

build_model('data/hobbies.classifier', corpus)

# We can now load our model from disk 
with open('data/hobbies.classifier', 'rb') as f:
    model = pickle.load(f)

# Let's create a normalization method for fetching URL content
# that our model expects, based on our methods above. 
def fetch(url):
    html = requests.get(url)
    text = extract(html.text)
    tokens = tokenize(text)
    tags = tag(tokens)
    lemmas = lemmatize(tags)
    return list(normalize(lemmas))

def predict(url):
    text = fetch(url)
    probs = zip(model.classes_, model.predict_proba([text])[0])
    label = model.predict([text])[0]
    
    print("y={}".format(label))
    for cls, prob in sorted(probs, key=lambda x: x[1]):
        print("  {}: {:0.3f}".format(cls, prob))

predict("http://minimalistbaker.com/5-ingredient-white-chocolate-truffles/")

