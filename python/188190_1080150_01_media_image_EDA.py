import pandas as pd

media_imgs_long = pd.read_csv("../assets/media_url_link_long.csv")

media_imgs_long.drop("Unnamed: 0", axis =1, inplace = True)

# for-loop to drop all "Human", "People", "Person" Label rows where the image 
# contains a celebrity.
for img in media_imgs_long["img"]:
    if len(media_imgs_long.loc[(media_imgs_long["img"] == img) & (media_imgs_long["type"]=="Celebrity")])>0:
        media_imgs_long = media_imgs_long.loc[~((media_imgs_long['img'] == img) 
                          & (media_imgs_long['label'].isin(['Human', 'People', 'Person'])))]

# for-loop to drop all "Label" rows below 90% confidence if there is a celebrity
for img in media_imgs_long["img"]:
    if len(media_imgs_long.loc[(media_imgs_long["img"] == img) & (media_imgs_long["type"]=="Celebrity")])>0:
        media_imgs_long = media_imgs_long.loc[~((media_imgs_long['img'] == img) 
                          & (media_imgs_long['type'].isin(['Label'])) & (media_imgs_long['confidence']<90))]

# for loop to drop all "Label", "Sticker", "Text" label rows where image contains text.
for img in media_imgs_long["img"]:
    if len(media_imgs_long.loc[(media_imgs_long["img"] == img) & (media_imgs_long["type"]=="Text")])>0:
        media_imgs_long = media_imgs_long.loc[~((media_imgs_long['img'] == img) 
                          & (media_imgs_long['label'].isin(['Label', 'Sticker', 'Text'])))]

import numpy as np

media_new = []
for img in media_imgs_long['img'].unique():
    img_dict = {'img': img}
    if len(media_imgs_long[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Label')])>0:
        img_dict['label'] = ' '.join(media_imgs_long.loc[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Label'), 'label'].tolist())
    else:
        img_dict['label'] = None
    if len(media_imgs_long[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Text')])>0:
        text = [str(detected_text) 
                for detected_text in media_imgs_long.loc[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Text'), 'label'].tolist()]
        img_dict['text'] = ' '.join(text)
    else:
        img_dict['text'] = None
    img_dict['celebrity'] = len(media_imgs_long[(media_imgs_long['img']==img) & (media_imgs_long['type']=='Celebrity')])>0
    media_new.append(img_dict)
media_new_df = pd.DataFrame(media_new)

media_new_df

media_new_df["text"] = [False if x == None else True for x in media_new_df["text"]]

media_new_df.to_csv("01_media_text_data.csv")

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

doc_set = media_new_df.loc[:,["img", "label"]]

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

ldamodel = models.ldamodel.LdaModel.load('labels_lda.model')

doc_set.reset_index(inplace=True)
doc_label_topic = []
for i, text in enumerate(corpus):
    topics = sorted(ldamodel[text], key=lambda x: -x[1])
    doc_label_topic.append({'img': doc_set['img'][i], 'label_topic': topics[0][0], 'label_topic_prob': topics[0][1]})
doc_label_topic_df = pd.DataFrame(doc_label_topic)

doc_label_topic_df.head()

media_new_df = media_new_df.merge(doc_label_topic_df, on='img', how='left')
media_new_df.head()

media_new_df = media_new_df.join(pd.get_dummies(media_new_df["label_topic"]))

media_new_df.drop(["label", "label_topic", "label_topic_prob"], axis =1, inplace = True)

media_new_df.to_csv("../assets/02_media_text_data.csv")

## some of those topic fitting percentages looked kind of low. Let's try to do topic modelling again. 

ldamodel2=gensim.models.ldamodel.LdaModel(corpus, num_topics=14, id2word = dictionary, passes=20)

#lda model 1
doc_set.reset_index(inplace=True)
doc_label_topic = []
for i, text in enumerate(corpus):
    topics = sorted(ldamodel[text], key=lambda x: -x[1])
    doc_label_topic.append({'img': doc_set['img'][i], 'label_topic': topics[0][0], 'label_topic_prob': topics[0][1]})
doc_label_topic_df = pd.DataFrame(doc_label_topic)

ldamodel[corpus[0]]

#might be worth while to create one LDA model on all the images.

doc_label_topic_df.head()

media_new_df = media_new_df.merge(doc_label_topic_df, on='img', how='left')
media_new_df.head()

media_new_df.to_csv("media_text_data.csv")



