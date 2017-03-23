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
import matplotlib.pyplot as plt
import numpy as np


def plotFeatures(hashTagName, labels, predictors, columnNum, xlabels):
    for i in range(3):
            plt.gca().scatter(labels, predictors[:,columnNum[i]], color='blue')
            plt.xlabel(xlabels[i])
            plt.ylabel("Tweets for next hour")
            plt.draw()
            imageName = hashTagName + "_best_feature_"+str(i+1)+".png"
            plt.savefig(imageName)
    plt.close()        

def fitModel(hashTagName, labels, predictors, predictorNum):
    predictors = np.transpose(np.asarray([predictors[:,predictorNum[0]],predictors[:,predictorNum[1]],predictors[:,predictorNum[2]],predictors[:,predictorNum[3]],predictors[:,predictorNum[4]],predictors[:,predictorNum[5]]]))
    model = sm.OLS(labels, predictors)
    results = model.fit()
    with open("problem_3_best_result"+hashTagName+".txt", 'wb') as fp:
        print >>fp, results.summary()


def getLablesPredictors(featuresList):
    start_time = min(featuresList.keys()) 
    end_time = max(featuresList.keys())    
    predictors = []
    labels = []
    cur_hour = start_time
    while cur_hour <= end_time: 
        next_hour_tweet_count = 0 
        next_hour = cur_hour+timedelta(hours=1) 
        if next_hour in featuresList:
            next_hour_tweet_count = featuresList[next_hour]['tweets_count'] 
        if cur_hour in featuresList:
            predictors.append(featuresList[cur_hour].values()) 
            labels.append([next_hour_tweet_count])
        else: 
            temp = {'tweets_count':0, 'retweets_count':0, 'ranking_score':0, 'impression_count':0,  'followers_count':0, 'max_followers':0, 'total_favorite_count':0, 'time':cur_hour.hour, 'total_user_mentions':0, 'long_tweet_count':0,'total_listed_count':0} 
            predictors.append(temp.values())
            labels.append([next_hour_tweet_count])
        cur_hour = next_hour
    return predictors, labels


def retrieveFeatures(filename):
    st = datetime.datetime(2017,03,20) 
    et = datetime.datetime(2001,03,20) 
    hourlyFeatures = {}
    uniqueUsers = {} 
    with open(filename) as tweet_data: 
        for individual_tweet in tweet_data: 
            tweetData = json.loads(individual_tweet) 
            iTime = tweetData["firstpost_date"] 
            iTime = datetime.datetime.fromtimestamp(iTime) 
            mTime = datetime.datetime(iTime.year, iTime.month, iTime.day, iTime.hour, 0, 0)
            mTime = unicode(mTime)
            tweetUserId = tweetData["tweet"]["user"]["id"] 
            retweet_count = tweetData["metrics"]["citations"]["total"]
            followers_count = tweetData["author"]["followers"]
            user_mentions = len(tweetData["tweet"]["entities"]["user_mentions"])
            listed_count = tweetData["tweet"]["user"]["listed_count"]
            if(listed_count == None):
                listed_count = 0
            favorite_count = tweetData["tweet"]["favorite_count"]
            ranking_score = tweetData["metrics"]["ranking_score"]
            title_Text = tweetData["title"]
            impression_count = tweetData["metrics"]["impressions"]
            if mTime not in hourlyFeatures:
                hourlyFeatures[mTime] = {'tweets_count':0, 'retweets_count':0, 'ranking_score':0, 'impression_count':0, 'followers_count':0, 'max_followers':0, 'total_favorite_count':0, 'time':-1,'total_user_mentions':0,
                'long_tweet_count':0,'total_listed_count':0}
                uniqueUsers[mTime] = Set([])
            hourlyFeatures[mTime]['tweets_count'] += 1
            hourlyFeatures[mTime]['retweets_count'] += retweet_count
            hourlyFeatures[mTime]['ranking_score'] += ranking_score
            hourlyFeatures[mTime]['impression_count'] += impression_count
            hourlyFeatures[mTime]['total_favorite_count'] += favorite_count
            hourlyFeatures[mTime]['time'] = iTime.hour
            hourlyFeatures[mTime]['total_user_mentions'] += user_mentions
            
            if len(title_Text) > 100:
                hourlyFeatures[mTime]['long_tweet_count'] += 1
            
            if tweetUserId not in uniqueUsers[mTime]:
                uniqueUsers[mTime].add(tweetUserId)
                hourlyFeatures[mTime]['followers_count'] += followers_count
                hourlyFeatures[mTime]['total_listed_count'] += listed_count
                if followers_count > hourlyFeatures[mTime]['max_followers']:
                    hourlyFeatures[mTime]['max_followers'] = followers_count
            if iTime < st:
                st = iTime
            if iTime > et:
                et = iTime
    return hourlyFeatures

dir = os.path.dirname(__file__)
folderName =  os.path.join(dir, "tweet_data")     
fileList =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

for i in range(5,len(fileList)):  
    print i
    filename = os.path.join(folderName, fileList[i])
    hourlyFeatures = retrieveFeatures(filename)
    modifiedHourlyFeatures = {}
    for time_value in hourlyFeatures:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = hourlyFeatures[time_value]
        modifiedHourlyFeatures[cur_hour] = features
    predictors, labels = getLablesPredictors(modifiedHourlyFeatures)

    predictors = sm.add_constant(predictors)    
    model = sm.OLS(labels, predictors)
    results = model.fit()
    with open("linear_regression_problem_3_result"+hashtag_list[i]+".txt", 'wb') as fp:
        print >>fp, results.summary()

    if(i==0):   #gohawks
        columnNum = [5, 6, 2]
        predictorNum = [1,2,5,6,4,8]
        xlabels = ["Feature : Individual listed frequency", "Feature : Individual mention frequency", "Feature : Ranking score"]
        plotFeatures(hashtag_list[i], labels, predictors, columnNum, xlabels) 
        fitModel(hashtag_list[i], labels, predictors, predictorNum)
    elif(i==1): #gopatriots
        columnNum = [2, 3, 9]
        predictorNum = [2, 7,9,1,4,3]
        xlabels = ["Feature : Ranking Score", "Feature : Impression Count", "Feature : Number of long tweets"]
        plotFeatures(hashtag_list[i], labels, predictors, columnNum, xlabels) 
        fitModel(hashtag_list[i], labels, predictors, predictorNum)
    elif(i==2): #nfl
        columnNum = [5, 11, 4]
        predictorNum = [5,11,4,1,7,6]
        xlabels = ["Feature : Individual listed frequency", "Feature : Number of maximum followers", "Feature : Number of tweets"]
        plotFeatures(hashtag_list[i], labels, predictors, columnNum, xlabels) 
        fitModel(hashtag_list[i], labels, predictors, predictorNum)
    elif(i==3): #patriots
        columnNum = [6, 2, 7]
        predictorNum = [6,2,7,4,11,9]
        xlabels = ["Feature : Individual mention frequency", "Feature : Ranking Score", "Feature : Total follower count"]
        plotFeatures(hashtag_list[i], labels, predictors, columnNum, xlabels) 
        fitModel(hashtag_list[i], labels, predictors, predictorNum)
    elif(i==4): #sb49
        columnNum = [2, 8, 7]
        predictorNum = [2,8,7,4,11,1]
        xlabels = ["Feature : Ranking score", "Feature : Number of retweets", "Feature : Total follower count"]
        plotFeatures(hashtag_list[i], labels, predictors, columnNum, xlabels) 
        fitModel(hashtag_list[i], labels, predictors, predictorNum)
    elif(i==5): #superbowl
        columnNum = [5, 4, 8]
        predictorNum = [5,4,8,6,1,2]
        xlabels = ["Feature : Individual listed frequency", "Feature : Number of tweets", "Feature : Number of retweets"]
        plotFeatures(hashtag_list[i], labels, predictors, columnNum, xlabels) 
        fitModel(hashtag_list[i], labels, predictors, predictorNum)