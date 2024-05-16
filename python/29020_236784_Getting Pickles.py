cd /Users/Simon/Topic-Ontology

import pandas as pd
data=pd.read_pickle('en_wikipedia_titles.pkl_df_500001_nlp.pkl')

def include_links_in_BOW(data, weight):
    BOW=data['proctext']
    for i in range(data.shape[0]):
        for j in range(weight):
            BOW[i]+=' '
            BOW[i]+=re.sub('_', ' ', data['proclinks'][i]) #don't worry about dashes b/c these are removed by tfidf 
        if i%1000==0:
            print(i)
    return BOW
    

import re

weighted_text=include_links_in_BOW(data, 2)

weighted_text[12407]

import pickle

pickle.dump(weighted_text, open('BagOfWords.pkl','wb'))

x=pickle.load(open('BagOfWords.pkl', 'rb'))

x[12407]

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(min_df=.008, stop_words='english')

matrix=tfidf.fit_transform(weighted_text)

len(tfidf.get_feature_names())

from sklearn.decomposition import PCA

pca=PCA(n_components=400)

x=pca.fit_transform(matrix.toarray())

y=pca.components_

len(y)

pickle.dump(matrix, open('Tfidf_Matrix.pkl','wb'))

pickle.dump(tfidf.get_feature_names(), open('Feature_List.pkl','wb'))

x.shape

pickle.dump(x, open('PCA_matrix.pkl', 'wb'))

pickle.dump(pca.components_, open('PCA_components.pkl', 'wb'))

a=pickle.load(open('Tfidf_Matrix.pkl', 'rb'))

link_BOW=[re.sub('-', '_', links) for links in data['proclinks']]

for i in range(data.shape[0]):
    if link_BOW[i].find('kuwait_national_under')>=0:
        print(i)

link_BOW[12407]

tfidf_links=TfidfVectorizer(min_df=.002, stop_words='english')

link_tfidf=tfidf_links.fit_transform(link_BOW)

len(tfidf_links.get_feature_names())

pickle.dump(link_tfidf, open('Corrected_Link_Tfidf_Matrix.pkl','wb'))
pickle.dump(tfidf_links.get_feature_names(), open('Corrected_Link_Feature_List.pkl','wb'))

link_pca=PCA(n_components=400)

link_pca_matrix=link_pca.fit_transform(link_tfidf.toarray())

pickle.dump(link_pca_matrix, open('Corrected_Link_PCA_matrix.pkl','wb'))
pickle.dump(link_pca.components_, open('Corrected_Link_PCA_components.pkl', 'wb'))

pickle.dump(link_BOW, open('Corrected_Link_BagOfWords.pkl','wb'))

x=['bob-jack jill', 'jill']
tfidf_test=TfidfVectorizer()
tfidf_test.fit_transform(x)
tfidf_test.get_feature_names()



