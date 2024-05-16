import pymongo
from pymongo import MongoClient
from nltk.corpus import stopwords
import string,logging,re
import pandas as pd
import gensim
import statsmodels.api as sm
import statsmodels.formula.api as smf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
c=MongoClient()
tweets_data=c.twitter.tweets
data=[x['text'] for x in tweets_data.find({'user.name':'Donald J. Trump'},{'text':1, '_id':0})]

table = str.maketrans({key: None for key in string.punctuation})
tokenized_tweets=list()
for d in data:
     text=[word for word in d.lower().split() if word not in stopwords.words("english")]
     text=[t.translate(table) for t in text]
     clean_tokens=[]
     for token in text:
        if re.search('[a-zA-Z]', token) and len(token)>1 and '@' not in token and token!='amp' and 'https' not in token:
            clean_tokens.append(token)
     tokenized_tweets.append(clean_tokens)
tag_tokenized=[gensim.models.doc2vec.TaggedDocument(tokenized_tweets[i],[i]) for i in range(len(tokenized_tweets))]
tag_tokenized[10]

print(data[:5])
print(len(data))
print ('\n')
print(tokenized_tweets[:5])
print(len(tokenized_tweets))

model = gensim.models.doc2vec.Doc2Vec(size=200, min_count=3, iter=200)
model.build_vocab(tag_tokenized)
model.train(tag_tokenized, total_examples=model.corpus_count, epochs=model.iter)
print(model.docvecs[10])

from sklearn.cluster import KMeans
num_clusters = 50
km = KMeans(n_clusters=num_clusters)
km.fit(model.docvecs)
clusters = km.labels_.tolist()

data_jsn=[x for x in tweets_data.find({},
                     {'favorite_count':1,'retweet_count':1,'created_at':1,'entities.hashtags':1,'entities.urls':1,
                      'entities.media':1,'_id':0})]

df=pd.io.json.json_normalize(data_jsn)

df['has_hashtags']=[len(a)>0 for a in df['entities.hashtags']]
df['has_urls']=[len(a)>0 for a in df['entities.urls']]
df['has_media']=1-df['entities.media'].isnull()
df['dow']=df.created_at.str[:3]
df['time']=df.created_at.str[11:13]
df['time_group']=df.time.astype('int64').mod(4)

df_clusters=pd.concat([pd.Series(clusters),df,pd.Series(data)],axis=1)

df_clusters.columns=['cluster']+list(df.columns)+['text']
df_clusters['cluster_cat']=pd.Categorical(df_clusters['cluster'])
df_clusters.head()

results_baseline = smf.ols('favorite_count ~ dow + time_group + has_media + has_hashtags + has_urls', data=df_clusters[(df_clusters.favorite_count>0)]).fit()
results_clusters = smf.ols('favorite_count ~ dow + time_group + has_media + has_hashtags + has_urls+cluster_cat', data=df_clusters[(df_clusters.favorite_count>0)]).fit()

print(results_baseline.summary())
print(results_clusters.summary())

#Cluster 15
#General topic is Fake News
[tweet for tweet in df_clusters[df_clusters.cluster_cat==15].text[:10]]

#Cluster 40
#General topic of NoSanctuaryForCriminalsAct
[tweet for tweet in df_clusters[df_clusters.cluster_cat==40].text[:10]]



