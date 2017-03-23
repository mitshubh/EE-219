# -*- coding: utf-8 -*-
"""
@author: swati.arora
"""

import os
import json
import datetime
from sets import Set
from datetime import timedelta
import os.path
import statsmodels.api as sm
import logging as logger

def getLablesPredictors(hourlyFeatures):
    start_time = min(hourlyFeatures.keys()) 
    end_time = max(hourlyFeatures.keys())    
    predictors = []
    labels = []
    cur_hour = start_time
    while cur_hour <= end_time: 
        next_hour_tweet_count = 0 
        next_hour = cur_hour+timedelta(hours=1) 
        if next_hour in hourlyFeatures:
            next_hour_tweet_count = hourlyFeatures[next_hour]['tweets_count'] 
        if cur_hour in hourlyFeatures:
            predictors.append(hourlyFeatures[cur_hour].values()) 
            labels.append([next_hour_tweet_count])
        else: 
            temp = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':cur_hour.hour}    
            predictors.append(temp.values())
            labels.append([next_hour_tweet_count])
        cur_hour = next_hour
    return predictors, labels

def retrieveFeatures(fileName):
    st = datetime.datetime(2017,03,20) 
    et = datetime.datetime(2001,03,20) 
    hourlyFeatures = {}
    hourlyUsers = {} 
    with open(fileName) as tweetData:
        for everyTweet in tweetData: 
            tweetData = json.loads(everyTweet) 
            individual_time = tweetData["firstpost_date"] 
            individual_time = datetime.datetime.fromtimestamp(individual_time) 
            modified_time = datetime.datetime(individual_time.year, individual_time.month, individual_time.day, individual_time.hour, 0, 0)
            modified_time = unicode(modified_time)
            tweetUserId = tweetData["tweet"]["user"]["id"] 
            retweetsCount = tweetData["metrics"]["citations"]["total"]
            followerCount = tweetData["author"]["followers"]
            if modified_time not in hourlyFeatures:
                hourlyFeatures[modified_time] = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':-1}
                hourlyUsers[modified_time] = Set([])
            hourlyFeatures[modified_time]['tweets_count'] += 1
            hourlyFeatures[modified_time]['retweets_count'] += retweetsCount
            hourlyFeatures[modified_time]['time'] = individual_time.hour
            if tweetUserId not in hourlyUsers[modified_time]: #If a user is not added, then add it
                hourlyUsers[modified_time].add(tweetUserId)
                hourlyFeatures[modified_time]['followers_count'] += followerCount
                if followerCount > hourlyFeatures[modified_time]['max_followers']:
                    hourlyFeatures[modified_time]['max_followers'] = followerCount
            if individual_time < st:
                st = individual_time
            if individual_time > et:
                et = individual_time
    return hourlyFeatures
  
dir = os.path.dirname(__file__)
folderName =  os.path.join(dir, "tweet_data")  
fileList =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtagList = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

for i in range(len(fileList)):   
    filename = os.path.join(folderName, fileList[i])
    hourlyFeaturesVals = retrieveFeatures(filename)
    modified_hourwise_features = {}
    for time_value in hourlyFeaturesVals:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        featuresVals = hourlyFeaturesVals[time_value]
        modified_hourwise_features[cur_hour] = featuresVals
    predictors, labels = getLablesPredictors(modified_hourwise_features)
    predictors = sm.add_constant(predictors)
    model = sm.OLS(labels, predictors)
    results = model.fit()    
    logger.info("======================================================================")
    logger.info('Parameters: {}'.format(results.params))
    logger.info('Standard errors: {}'.format(results.bse))
    logger.info('p values: {}'.format(results.pvalues))
    logger.info('t values: {}'.format(results.tvalues))
    logger.info('Accuracy: {}'.format(results.rsquared * 100))
    logger.info("======================================================================")
    with open("linear_regression_result_problem_2_"+hashtagList[i]+".txt", 'wb') as fp:
        print >>fp, results.summary()