import pandas as pd
from textblob import TextBlob
import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('dark')
sns.set_context('talk')
sns.set_style('white')

data = pd.read_pickle('../priv/pkl/03_cellartracker_dot_com_data.pkl')

data = data.loc[data.review_text.isnull().pipe(np.invert)]

data['review_points'] = data.review_points.astype(float)

data.head(2)

data['textblob'] = data.review_text.apply(lambda x: TextBlob(x))

data2 = data[:2000].copy()

data2['polarity'] = data2.textblob.apply(lambda x: x.sentiment.polarity)
data2['subjectivity'] = data2.textblob.apply(lambda x: x.sentiment.subjectivity)

data2.dtypes

ax = data2.plot('review_points', 'polarity', marker='o',ls='')

ax = data2.plot('review_points', 'subjectivity', marker='o',ls='')

