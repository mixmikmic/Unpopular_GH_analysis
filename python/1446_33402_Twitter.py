# If needed
# !pip install tweepy
# !pip install textblob
# !pip install nltk
# 2wEURk users, add "--user"

# If needed
# import nltk
# nltk.download()  # Select twitter_samples under tab 'Corpora'

# Imports always goes on top
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import twitter_samples
import json
import random

# Don't want this in GitHub or show on the slides
import twitter_credentials

# We'll train a classifier on the NLTK twitter samples
# This takes some time, so do only once per session

# List of 2-tuples, with each 2-tuple a list of strings and a label  
train = []

# First the negs
for tokens in twitter_samples.tokenized('negative_tweets.json'):
    train.append((tokens, 'neg'))
    
# First the poss
for tokens in twitter_samples.tokenized('positive_tweets.json'):
    train.append((tokens, 'pos'))

# Take a subset, speed up training
random.shuffle(train)
train = train[0:200]

#print(train[0])
cl = NaiveBayesClassifier(train)

class Tweet:
    """This class creates a tweet from a JSON string"""
    def __init__(self, data, cl):
        # Hint : print(self._tweet.keys()) for all keys in the tweet
        self._tweet = json.loads(data)
        self.blob1 = TextBlob(self._tweet["text"], classifier=cl)
        self.blob2 = TextBlob(self._tweet["text"], analyzer=NaiveBayesAnalyzer())
        
    def print_tweet(self):
        print()
        print("-" * 80)
        print(self._tweet["id_str"], self._tweet["created_at"])
        print(self._tweet["text"])
    
    def print_language(self):
        print("language", self.blob1.detect_language())
        
    def print_sentiment(self):
        print("sentiment", self.blob1.classify())
        print(self.blob2.sentiment)

class MyListener(StreamListener):
    """Listener class that processes a Twitter Stream"""
    def __init__(self, max_count, cl):
        self.max_count = max_count
        self.count = 0
        self.cl = cl
    
    def on_data(self, data):
        self.tweet = Tweet(data, cl)
        self.tweet.print_tweet()
        self.tweet.print_language()
        self.tweet.print_sentiment()
                
        self.count += 1
        if self.count >= self.max_count:
            return False
        return True

# Create the auth object
# https://www.slickremix.com/docs/how-to-get-api-keys-and-tokens-for-twitter/
auth = OAuthHandler(twitter_credentials.consumer_key, twitter_credentials.consumer_secret)
auth.set_access_token(twitter_credentials.access_token, twitter_credentials.access_token_secret)

# Create a listener, define max tweets we'll process, pass the classifier
mylistener = MyListener(10, cl)

# Create a stream, and use the listener to process the data
mystream = Stream(auth, listener=mylistener)

# Creating a list of keywords to search the Tweets
keywords = ['Python', 'Jupyter', 'eur.nl']

# Start the stream, based on the keyword-list
mystream.filter(track = keywords)

# Disconnects the streaming data
mystream.disconnect()

