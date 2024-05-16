import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')

from unidecode import unidecode
from nltk.corpus import stopwords

import nltk

# file is UTF-8 encoding
with open('wife.txt') as f:
     line = f.read()

line

type(line)

line = line.decode('utf-8')

type(line)

line

line = line.encode("ascii", "ignore").replace('\n', ' ')
line

s = ''
for c in line:
     s += c if c.isupper() else ' '
s

import re
s = re.sub("[^A-Z]", " ", line)
s

from collections import Counter
c = Counter(s.split())
c.most_common()

main_chars = ["DOGGY", "PETERINE", "PUCK", "THORGOLF", "STARLIGHT", "AGATHA", "ALLONDRA",
              "VITO", "PAMELA", "YUN", "ELEANOR", "CLYDE", "TRAUT", "SMACKERS", "TIPTANNER",
              "IVY", "IRVING", "RODNEY", "ROSEMARY", "SIMONE", "WILLARD", "CHANTILLY",
              "PINEAPPLE", "STOVE", "REVEREND"]

wife_corpus = nltk.Text(line.split())

fig, ax = plt.subplots(figsize=(12, 10))
wife_corpus.dispersion_plot(sorted(main_chars))
fig.savefig('dispersion_plot_wife.jpg')

main_chars_lower = set(item.lower() for item in main_chars)
stops = stopwords.words("english")
letters_only = re.sub("[^a-zA-Z]", " ", line)
words = letters_only.lower().split()
h = [word for word in words if all([word not in stops, word not in main_chars_lower])]

fig, ax = plt.subplots(figsize=(12, 10))
fdist = nltk.FreqDist(h)
fdist.plot(50, cumulative=False)

