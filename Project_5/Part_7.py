from helper import *
from celebs import *
from ads import *
from topics import *
import json
import pandas as pd

inputFile = 'tweets_#superbowl1.txt'
geoLat = []
geoLong = []
tweets = open('tweet_data/' + inputFile, 'rb')
tweetBag = []
tweetsPerHour = []
win = 1
t1 = json.loads(tweets.readline())
start = t1.get('firstpost_date')
wnEnd = start + win * 3600
tweetCount = len(tweets.readlines())
tweets.seek(0, 0)

for count, tweet in enumerate(tweets):
    tweetJSON = json.loads(tweet)
    tweetTextData = tweetJSON.get("tweet").get("text")
    endTime = tweetJSON.get('firstpost_date')
    geolocation = tweetJSON.get('tweet').get("coordinates")
    if geolocation is not None:
        geoLat.append(geolocation['coordinates'][0])
        geoLong.append(geolocation['coordinates'][1])
    if endTime < wnEnd:
        tweetsPerHour.append(tweetTextData)
    else:
        word_list, other_hash_tags, key_words, bigrams_counter = preprocess_data(tweetsPerHour)
        ads_df = get_advertisements(other_hash_tags, key_words, bigrams_counter)
        celeb_df = get_celebrities(other_hash_tags, key_words)
        perform_modelling(celeb_df, ads_df, key_words)
        tweetBag.append(key_words)
        win += 1
        tweetsPerHour = []
        wnEnd = start + win * 3600

adTimeSeries(start)
topicsTimeSeries(start)