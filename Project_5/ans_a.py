# -*- coding: utf-8 -*-
"""
@author : swati.arora

"""

import os
import json
import datetime
from sets import Set
from datetime import timedelta
import matplotlib.pyplot as plt


def calTagInfo(filename):
    totalFollowers = 0.0
    retweetsCount = 0.0 
    totalTweets = 0.0
    totalHours = 0.0
    uniqueUsers = Set([]) 
    hour = {} 
    st = datetime.datetime(2017,03,20) 
    et = datetime.datetime(2001,03,20)
    with open(filename) as tweetData: 
        for everyTweet in tweetData: 
            tweetData = json.loads(everyTweet) 
            tweetUserId = tweetData["tweet"]["user"]["id"] 
            if tweetUserId not in uniqueUsers: 
                totalFollowers += tweetData["author"]["followers"]
                uniqueUsers.add(tweetUserId)
            retweetsCount += tweetData["metrics"]["citations"]["total"] 
            individual_time = tweetData["firstpost_date"] 
            individual_time = datetime.datetime.fromtimestamp(individual_time) 
            if individual_time < st:
                st = individual_time
            if individual_time > et:
                et = individual_time
            totalTweets = totalTweets + 1 
            modified_time = datetime.datetime(individual_time.year, individual_time.month, individual_time.day, individual_time.hour, 0, 0)
            modified_time = unicode(modified_time)
            if modified_time not in hour:
                hour[modified_time] = {'hour_tweets_count':0}
            hour[modified_time]['hour_tweets_count'] += 1 
    totalHours = int((et - st).total_seconds()/3600 + 0.5)  
    return totalFollowers, retweetsCount, totalTweets, totalHours, len(uniqueUsers), hour


def drawHistogram(filename, hour):
    tweetTimeStart = min(hour.keys())
    tweetTimeEnd = max(hour.keys())
    perHourTweets = []
    current_time = tweetTimeStart
    while current_time <= tweetTimeEnd:
        if current_time in hour:
            perHourTweets.append(hour[current_time]["hour_tweets_count"])
        else:
            perHourTweets.append(0)
        current_time += timedelta(hours=1)        
    plt.figure(figsize=(15, 6))
    plt.title("Number of tweets per hour for " + filename)
    plt.ylabel("Number of Tweets")
    plt.xlabel("Hours Elapsed") 
    plt.bar(range(len(perHourTweets)), perHourTweets)
    plt.show()
    
    
dir = os.path.dirname(__file__)
folderName =  os.path.join(dir, "tweet_data")
fileList =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtagList = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

for i in range(len(fileList)):
    filename = os.path.join(folderName, fileList[i])
    totalFollowers, retweetsCount, totalTweets, totalHours, total_users, hour = calTagInfo(filename)
    print "Avg. number of tweets/hour for",  hashtagList[i], "are : ", totalTweets/totalHours
    print "Avg. number of followers of person tweeting for",  hashtagList[i], "are : ", totalFollowers/total_users
    print "Avg. number of retweets for", hashtagList[i], "are : ", retweetsCount/totalTweets
    modified_hour = {}
    for time_value in hour:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = hour[time_value]
        modified_hour[cur_hour] = features
    drawHistogram(hashtagList[i], modified_hour)

