import pandas as pd
pacs_scraped = pd.DataFrame.from_csv('public.raw_committees_scraped.csv')  # id
pacs = pd.DataFrame.from_csv('public.raw_committees.csv')  # no ID that I can find
candidates = pd.DataFrame.from_csv('public.raw_candidate_filings.csv')  # id_nmbr
print(pacs_scraped.info())
print(candidates.info())

import re
from itertools import product
regex = re.compile(r'(\b|_|^)[Ii][Dd](\b|_|$)')
pac_id_cols = [col for col in pacs.columns if regex.search(col)]
print(pac_id_cols)
pac_scraped_id_cols = [col for col in pacs_scraped.columns if regex.search(col)]
print(pac_scraped_id_cols)
candidate_id_cols = [col for col in candidates.columns if regex.search(col)]
print(candidate_id_cols)
trans = pd.DataFrame.from_csv('public.raw_committee_transactions_ammended_transactions.csv')
trans_id_cols = [col for col in trans.columns if regex.search(col)]
print(trans_id_cols)
tables = [('pac', pacs, pac_id_cols), ('pac_scraped', pacs_scraped, pac_scraped_id_cols), ('candidate', candidates, candidate_id_cols), ('trans', trans, trans_id_cols)]
graph = []
for ((n1, df1, cols1), (n2, df2, cols2)) in product(tables, tables):
    if n1 == n2:
        continue
    for col1 in cols1:
        for col2 in cols2:
            s1 = set(df1[col1].unique())
            s2 = set(df2[col2].unique())
            similarity = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
            print('{}.{} -- {:.3} -- {}.{}'.format(n1, col1, similarity, n2, col2 ))
            graph += [(n1, col1, similarity, n2, col2)]
graph = pd.DataFrame(sorted(graph, key=lambda x:x[2]), columns=['table1', 'column1', 'similarity', 'table2', 'column2'])
print(graph)

print(pacs_scraped.index.dtype)
print(pacs.index.dtype)

trans = pd.DataFrame.from_csv('public.raw_committee_transactions_ammended_transactions.csv')
trans.describe()

filtered_trans = []
for id in trans.original_id.unique():
    rows = sorted(trans[trans.original_id == id].iterrows(), key=lambda x:x[1].attest_date, reverse=True)
    filtered_trans += [rows[0][1]]
filtered_trans = pd.DataFrame(filtered_trans)
print(len(trans) / float(len(filtered_trans)))
print(filtered_trans.describe())

df = filtered_trans
filer_sums = df.groupby('filer_id').amount.sum()
print(pacs_scraped.columns)
print(df.columns)
for (filer_id, amount) in sorted(filer_sums.iteritems(), key=lambda x:x[1], reverse=True):
    names = pacs_scraped[pacs_scraped.id == filer_id].index.values
    print('{}\t{}\t{}'.format(filer_id, names[0][:40] if len(names) else '', amount))

import matplotlib
get_ipython().magic('matplotlib inline')
np = pd.np
np.norm = np.linalg.norm
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

df = pacs_scraped
names = df.index.values
corpus = [' '.join(str(f) for f in fields) for fields in zip(*[df[col] for col in df.columns if df[col].dtype == pd.np.dtype('O')])]
print(corpus[:3])
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english')
tfidf = vectorizer.fit_transform(corpus)
cov = tfidf * tfidf.T
cov[0:]

