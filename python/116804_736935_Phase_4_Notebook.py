get_ipython().magic('run helper_functions.py')
get_ipython().magic('run df_functions.py')
import string
import nltk
import spacy
nlp = spacy.load('en')
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.cluster import KMeans

gabr_tweets = unpickle_object("gabr_ibrahim_tweets_LDA_Complete.pkl")

gabr_tweets[0]['gabr_ibrahim'].keys() #just to refresh our mind of the keys in the sub-dictionary

temp_gabr_df = pd.DataFrame.from_dict(gabr_tweets[0], orient="index")

temp_gabr_df = filtration(temp_gabr_df, "content")

gabr_tweets_filtered_1 = dataframe_to_dict(temp_gabr_df)

clean_tweet_list = []
totalvocab_tokenized = []
totalvocab_stemmed = []


for tweet in gabr_tweets_filtered_1[0]['gabr_ibrahim']['content']:
    clean_tweet = ""
    to_process = nlp(tweet)
    
    for token in to_process:
        if token.is_space:
            continue
        elif token.is_punct:
            continue
        elif token.is_stop:
            continue
        elif token.is_digit:
            continue
        elif len(token) == 1:
            continue
        elif len(token) == 2:
            continue
        else:
            clean_tweet += str(token.lemma_) + ' '
            totalvocab_tokenized.append(str(token.lemma_))
            totalvocab_stemmed.append(str(token.lemma_))
            
    clean_tweet_list.append(clean_tweet)

#just going to add this to the dictionary so we can do the second round of filtration
gabr_tweets_filtered_1[0]['gabr_ibrahim']['temp_tfidf'] = clean_tweet_list

temp_gabr_df = pd.DataFrame.from_dict(gabr_tweets_filtered_1[0], orient='index')

temp_gabr_df = filtration(temp_gabr_df, 'temp_tfidf')

gabr_tweets_filtered_2 = dataframe_to_dict(temp_gabr_df)

clean_tweet_list = gabr_tweets_filtered_2[0]['gabr_ibrahim']["temp_tfidf"]
del gabr_tweets_filtered_2[0]["gabr_ibrahim"]["temp_tfidf"] # we will add back TF-IDF analysis later!

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('There are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_features=200000, stop_words='english', ngram_range=(0,2))

tfidf_matrix = tfidf_vectorizer.fit_transform(clean_tweet_list) #fit the vectorizer to synopses

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

num_clusters = 20

km = KMeans(n_clusters=num_clusters, n_jobs=-1, random_state=200)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

cluster_dict = dict()
for i in range(num_clusters):
    for ind in order_centroids[i, :20]: #replace 6 with n words per cluster
        word = str(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
        if i not in cluster_dict:
            cluster_dict[i] = [word]
        else:
            cluster_dict[i].append(word)

cluster_dict.keys() #here we see all 20 clusters.

cluster_dict[0] #words in cluster 1

cluster_dict[1] #words in cluster 2

cluster_dict[2] #words in cluster 3

#Now lets make our tfidf Counter!
cluster_values = []

for k, v in cluster_dict.items():
    cluster_values.extend(v)

counter_gabr_tfidf = Counter(cluster_values)

counter_gabr_tfidf

gabr_tweets_filtered_2[0]['gabr_ibrahim']["tfid_counter"] = counter_gabr_tfidf

gabr_tfidf_counter = gabr_tweets_filtered_2[0]['gabr_ibrahim']["tfid_counter"]

gabr_lda_counter = gabr_tweets_filtered_2[0]['gabr_ibrahim']["LDA"]

gabr_tfidf_set = set()
gabr_lda_set = set()

for key, value in gabr_tfidf_counter.items():
    gabr_tfidf_set.add(key)

for key, value in gabr_lda_counter.items():
    gabr_lda_set.add(key)

intersection = gabr_tfidf_set.intersection(gabr_lda_set)

gabr_tweets_filtered_2[0]['gabr_ibrahim']["lda_tfid_intersection"] = intersection

pickle_object(gabr_tweets_filtered_2, "FINAL_GABR_DATABASE_LDA_TFIDF_VERIFIED")



