import tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Assumes open twitter API called api
# Returns json
def get_tweets(handle,tweet_count):
    if(tweet_count<=200):
        tw=api.user_timeline(screen_name = handle,count=tweet_count)
        tw_json=[e._json for e in tw]
        return (tw_json)

    tw_json=[]
    cur_max=999999999999999999
    loop_count=tweet_count-tweet_count%200
    for i in range(0,loop_count,200):
         tw=(api.user_timeline(screen_name = handle,count=200,max_id=cur_max))
         tw_json=tw_json+([e._json for e in tw])
         cur_max=tw_json[-1:][0]['id']
    tw=(api.user_timeline(screen_name = handle,count=tweet_count-loop_count,max_id=cur_max))
    tw_json=tw_json+([e._json for e in tw])
    return (tw_json)

tw=get_tweets('@RealDonaldTrump',3200)

import pymongo
from pymongo import MongoClient
c=MongoClient()
tweets=c.twitter.tweets

tweets.insert_many(tw)
tweets.find_one()





