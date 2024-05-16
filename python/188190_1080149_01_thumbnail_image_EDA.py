import pandas as pd

thumb_imgs_long = pd.read_csv("../assets/thumbnail_link_long.csv")

thumb_imgs_long.head()

df = pd.read_csv("../gitignore/newtweets_10percent.csv")

mapping_dict = pd.read_csv("../assets/mapping_dict_thumbnail.csv")

mapping_dict.columns = [["img", "link_thumbnail"]]

df.columns

df = df[["brand", "engagement", "impact", "timestamp", "favorite_count", "hashtags", "retweet_count", "link_thumbnail"]]

df.head()

thumb = thumb_imgs_long.drop("Unnamed: 0", 1)

thumb.head()

# for-loop to drop all "Human", "People", "Person" Label rows where the image 
# contains a celebrity.
for img in mapping_dict["img"]:
    if len(thumb.loc[(thumb["img"] == img) & (thumb["type"]=="Celebrity")])>0:
        thumb = thumb.loc[~((thumb['img'] == img) 
                          & (thumb['label'].isin(['Human', 'People', 'Person'])))]

# for-loop to drop all "Label" rows below 90% confidence if there is a celebrity
for img in mapping_dict["img"]:
    if len(thumb.loc[(thumb["img"] == img) & (thumb["type"]=="Celebrity")])>0:
        thumb = thumb.loc[~((thumb['img'] == img) 
                          & (thumb['type'].isin(['Label'])) & (thumb['confidence']<90))]

# for loop to drop all "Label", "Sticker", "Text" label rows where image contains text.
for img in mapping_dict["img"]:
    if len(thumb.loc[(thumb["img"] == img) & (thumb["type"]=="Text")])>0:
        thumb = thumb.loc[~((thumb['img'] == img) 
                          & (thumb['label'].isin(['Label', 'Sticker', 'Text'])))]

import numpy as np

thumb.head(10)

thumb_new = []
for img in thumb['img'].unique():
    img_dict = {'img': img}
    if len(thumb[(thumb['img']==img) & (thumb['type']=='Label')])>0:
        img_dict['label'] = ' '.join(thumb.loc[(thumb['img']==img) & (thumb['type']=='Label'), 'label'].tolist())
    else:
        img_dict['label'] = None
    if len(thumb[(thumb['img']==img) & (thumb['type']=='Text')])>0:
        text = [str(detected_text) 
                for detected_text in thumb.loc[(thumb['img']==img) & (thumb['type']=='Text'), 'label'].tolist()]
        img_dict['text'] = ' '.join(text)
    else:
        img_dict['text'] = None
    img_dict['celebrity'] = len(thumb[(thumb['img']==img) & (thumb['type']=='Celebrity')])>0
    thumb_new.append(img_dict)
thumb_new_df = pd.DataFrame(thumb_new)

thumb_new_df["text"] = [False if x == None else True for x in thumb_new_df["text"]]

thumb_new_df.to_csv("01_thumb_text_data.csv")

import pandas as pd

#thumb_new_df = pd.read_csv("../assets/02_thumb_text_data.csv")

thumb_new_df.head()

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()  

doc_set = thumb_new_df.loc[:,["img", "label"]]

# compile sample documents into a list
doc_set.dropna(inplace=True)

texts = []

# loop through document list
for i in doc_set.label:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=14, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=14, num_words=5))

#this line is commented out so we don't re-save over our LDA model on images.
#ldamodel.save('labels_lda_14.model')

doc_set.reset_index(inplace=True)
doc_label_topic = []
for i, text in enumerate(corpus):
    topics = sorted(ldamodel[text], key=lambda x: -x[1])
    doc_label_topic.append({'img': doc_set['img'][i], 'label_topic': topics[0][0], 'label_topic_prob': topics[0][1]})
doc_label_topic_df = pd.DataFrame(doc_label_topic)

ldamodel[corpus[0]]

doc_label_topic_df.head()

thumb_new_df = thumb_new_df.merge(doc_label_topic_df, on='img', how='left')
thumb_new_df.head()

vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

pyLDAvis.display(vis)

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=25, id2word = dictionary, passes=20)

vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis)

#we picked 14 topics. 
thumb_new_df.head()

thumb_new_df.head(10)

thumb_new_df = thumb_new_df.join(pd.get_dummies(thumb_new_df["label_topic"]))

image_df = thumb_new_df.drop(["label", "label_topic"], axis =1)

image_df = image_df.drop("label_topic_prob", axis=1)

image_df.to_csv("../assets/02_thumb_data.csv")

image_df



