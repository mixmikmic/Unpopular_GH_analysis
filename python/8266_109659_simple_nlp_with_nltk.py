get_ipython().magic('matplotlib inline')
#%load_ext autoreload
#%autoreload 2
get_ipython().magic('reload_ext autoreload')
import numpy as np
import matplotlib.pyplot as plt
import math, sys, os
from numpy.random import randn
from sklearn.datasets import make_blobs

# setup pyspark for IPython_notebooks
spark_home = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))
execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))

import nltk
from nltk.book import *

text1.concordance("monstrous")

text1.similar("monstrous")

text2.common_contexts(["monstrous", "very"])

# text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

def lexical_diversity(text):
    return len(text) / len(set(text))

def percentage(count, total):
    return 100 * count / total

from BeautifulSoup import BeautifulSoup          # For processing HTML
#from BeautifulSoup import BeautifulStoneSoup     # For processing XML
#import BeautifulSoup                             # To get everything
from urllib import *

url = "http://www.gutenberg.org/files/2554/2554.txt"
raw = urlopen(url).read()
print type(raw)
print len(raw)
print raw[:75]

tokens = nltk.word_tokenize(raw)
print type(tokens)
print len(tokens)
print tokens[:10]

text = nltk.Text(tokens)
print type(text)
print text[1020:1060]
print text.collocations()

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = urlopen(url).read()
print html[:60]
soup = BeautifulSoup(html) # BeautifulSoup class
raw = soup.getText()
tokens = nltk.word_tokenize(raw)
print tokens[96:399]



