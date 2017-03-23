# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 01:51:42 2017

@author: amehrotra
"""

import os
import json
import datetime
from sets import Set
from datetime import timedelta
import os.path
import statsmodels.api as sm
from sklearn.cross_validation import KFold
import numpy as np


        
folder_name = "tweet_data"
train_files =  ["tweets_#gohawks.txt",
                "tweets_#gopatriots.txt",
                "tweets_#nfl.txt", 
                "tweets_#patriots.txt",
                "tweets_#sb49.txt", 
                "tweets_#superbowl.txt"]
                
hashtag_list = ["#gohawks",
                "#gopatriots",
                "#nfl",
                "#patriots",
                "#sb49", 
                "#superbowl"]
                
test_files = ["sample1_period1.txt",
              "sample2_period2.txt",
              "sample3_period3.txt",
              "sample4_period1.txt",
              "sample5_period1.txt",
              "sample6_period2.txt",
              "sample7_period3.txt",
              "sample8_period1.txt",
              "sample9_period2.txt",
              "sample10_period3.txt"]
              
def getLabelsMatrix(hourFeatures):
    predictors = []
    labels = []
    start = min(hourFeatures.keys()) 
    end = max(hourFeatures.keys())
    curr = start
    while curr <= end: 
        nxtHrTweetCnt = 0 
        next = curr+timedelta(hours=1)
        if next in hourFeatures:
            nxtHrTweetCnt = hourFeatures[next]['tweets_count'] 
        if curr in hourFeatures:
            predictors.append(hourFeatures[curr].values()) 
            labels.append([nxtHrTweetCnt])
        else: 
            temp = {'tweets_count':0, 'retweets_count':0, 'flwCt':0, 'max_followers':0, 'time':curr.hour,'avg_user_mention_count':0,
                'url':0,'avg_listed_count':0,'max_listed_count':0,'avg_favorite_count':0,'max_favorite_count':0,'sum_ranking_score':0,'total_verified_users':0,'user_count':0}    
            labels.append([nxtHrTweetCnt])            
            predictors.append(temp.values())          
        curr = next
    return predictors, labels

def getPeriodHourlyFeatures(file_id):
    hourFeatures = {}
    noOfUserPerHour = {}
    st = datetime.datetime(2015,2,1,8,0,0) 
    et = datetime.datetime(2015,2,1,20,0,0)
    
    fileName = os.path.join(folder_name, train_files[file_id])
    with open(fileName) as tweet_data: 
        for indvTweet in tweet_data:
            indvTweetData = json.loads(indvTweet) 
            indvTime = indvTweetData["firstpost_date"] 
            indvTime = datetime.datetime.fromtimestamp(indvTime) 
            modTime = datetime.datetime(indvTime.year, indvTime.month, indvTime.day, indvTime.hour, 0, 0)
            modTime = unicode(modTime)
            userId = indvTweetData["tweet"]["user"]["id"] 
            reCt = indvTweetData["metrics"]["citations"]["total"]
            followers_count = indvTweetData["author"]["followers"]      
            user_mention_count = len(indvTweetData["tweet"]["entities"]["user_mentions"])
            url = len(indvTweetData["tweet"]["entities"]["urls"])
            if url>0:
                url = 1 
            else:
                url = 0
            listedCt = indvTweetData["tweet"]["user"]["listed_count"]
            if(listedCt == None):
                listedCt = 0       
            fvCt = indvTweetData["tweet"]["favorite_count"]          
            rnkScr = indvTweetData["metrics"]["ranking_score"]   
            userVerified = indvTweetData["tweet"]["user"]["verified"]          
            if modTime not in hourFeatures:
                hourFeatures[modTime] = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':-1,'avg_user_mention_count':0,
                'url':0,'avg_listed_count':0,'max_listed_count':0,'avg_favorite_count':0,'max_favorite_count':0,'sum_ranking_score':0,'total_verified_users':0,'user_count':0}
                noOfUserPerHour[modTime] = Set([])
            hourFeatures[modTime]['tweets_count'] += 1
            hourFeatures[modTime]['retweets_count'] += reCt
            hourFeatures[modTime]['time'] = indvTime.hour
            hourFeatures[modTime]['avg_user_mention_count'] += user_mention_count
            hourFeatures[modTime]['url'] += url
            hourFeatures[modTime]['avg_favorite_count'] += fvCt
            if fvCt > hourFeatures[modTime]['max_favorite_count']:
                hourFeatures[modTime]['max_favorite_count'] = fvCt
            hourFeatures[modTime]['sum_ranking_score'] += rnkScr
            if userId not in noOfUserPerHour[modTime]:
                noOfUserPerHour[modTime].add(userId)
                hourFeatures[modTime]['followers_count'] += followers_count
                hourFeatures[modTime]['avg_listed_count'] += listedCt
                hourFeatures[modTime]['user_count'] += 1
                if followers_count > hourFeatures[modTime]['max_followers']:
                    hourFeatures[modTime]['max_followers'] = followers_count
                if listedCt > hourFeatures[modTime]['max_listed_count']:
                    hourFeatures[modTime]['max_listed_count'] = listedCt
                if  (userVerified):
                    hourFeatures[modTime]['total_verified_users'] += 1
    modHwFt = {}
    for time_value in hourFeatures:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = hourFeatures[time_value]
        modHwFt[cur_hour] = features    
    all_keys = modHwFt.keys()
    period1 = {}   
    period2 = {}
    period3 = {}    
    for key in all_keys:
        if(key < st):
            period1[key] = modHwFt[key]
        elif(key >= st and key <= et):
            period2[key] = modHwFt[key]
        else:
            period3[key] = modHwFt[key]
    return modHwFt, period1, period2, period3

def getHrlyFeat(file_name):
    
    hrFeatures = {}
    usrPerHr = {}
    with open(file_name) as tweet_data: 
        for indvTweet in tweet_data: 
            indvTweetDat = json.loads(indvTweet) 
            indvTime = indvTweetDat["firstpost_date"] 
            indvTime = datetime.datetime.fromtimestamp(indvTime) 
            modTime = datetime.datetime(indvTime.year, indvTime.month, indvTime.day, indvTime.hour, 0, 0)
            modTime = unicode(modTime)     
            userId = indvTweetDat["tweet"]["user"]["id"] 
            retweeCt = indvTweetDat["metrics"]["citations"]["total"]
            flwCt = indvTweetDat["author"]["followers"]     
            userMnCt = len(indvTweetDat["tweet"]["entities"]["user_mentions"])
            url = len(indvTweetDat["tweet"]["entities"]["urls"])
            if url>0:
                url = 1 
            else:
                url = 0
            lstCt = indvTweetDat["tweet"]["user"]["listed_count"]
            if(lstCt == None):
                lstCt = 0        
            fvCt = indvTweetDat["tweet"]["favorite_count"]           
            rnkScr = indvTweetDat["metrics"]["ranking_score"]          
            user_verified = indvTweetDat["tweet"]["user"]["verified"]           
            if modTime not in hrFeatures:
                hrFeatures[modTime] = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':-1,'avg_user_mention_count':0,
                'url':0,'avg_listed_count':0,'max_listed_count':0,'avg_favorite_count':0,'max_favorite_count':0,'sum_ranking_score':0,'total_verified_users':0,'user_count':0}
                usrPerHr[modTime] = Set([])
            hrFeatures[modTime]['tweets_count'] += 1
            hrFeatures[modTime]['retweets_count'] += retweeCt
            hrFeatures[modTime]['time'] = indvTime.hour
            hrFeatures[modTime]['avg_user_mention_count'] += userMnCt
            hrFeatures[modTime]['url'] += url
            hrFeatures[modTime]['avg_favorite_count'] += fvCt
            if fvCt > hrFeatures[modTime]['max_favorite_count']:
                hrFeatures[modTime]['max_favorite_count'] = fvCt
            hrFeatures[modTime]['sum_ranking_score'] += rnkScr
            if userId not in usrPerHr[modTime]:
                usrPerHr[modTime].add(userId)
                hrFeatures[modTime]['followers_count'] += flwCt
                hrFeatures[modTime]['avg_listed_count'] += lstCt
                hrFeatures[modTime]['user_count'] += 1
                if flwCt > hrFeatures[modTime]['max_followers']:
                    hrFeatures[modTime]['max_followers'] = flwCt
                if lstCt > hrFeatures[modTime]['max_listed_count']:
                    hrFeatures[modTime]['max_listed_count'] = lstCt
                if  (user_verified):
                    hrFeatures[modTime]['total_verified_users'] += 1
    modHrFeatures = {}
    for time_value in hrFeatures:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = hrFeatures[time_value]
        modHrFeatures[cur_hour] = features    
    return modHrFeatures


x,period1SB, period2SB, period3SB = getPeriodHourlyFeatures(5)    
x,period1Nfl, period2Nfl, period3Nfl = getPeriodHourlyFeatures(3)  
   
predict1SB, lb1SB = getLabelsMatrix(period1SB)
predict1SB = sm.add_constant(predict1SB)      
predict2SB, lb2SB = getLabelsMatrix(period2SB)
predict2SB = sm.add_constant(predict2SB)      
predict3SB, lb3SB = getLabelsMatrix(period3SB)
predict3SB = sm.add_constant(predict3SB) 
predict1Nfl, lb1Nfl = getLabelsMatrix(period1Nfl)
predict1Nfl = sm.add_constant(predict1Nfl)      
predict2Nfl, lb2Nfl = getLabelsMatrix(period2Nfl)
predict2Nfl = sm.add_constant(predict2Nfl)      
predict3Nfl, lb3Nfl = getLabelsMatrix(period3Nfl)
predict3Nfl = sm.add_constant(predict3Nfl) 
mdl2SB = sm.OLS(lb2SB, predict2SB)
res2SB = mdl2SB.fit()
mdl3SB = sm.OLS(lb3SB, predict3SB)
res3SB = mdl3SB.fit()
mdl1SB = sm.OLS(lb1SB, predict1SB)
res1SB = mdl1SB.fit()
mdl3Nfl = sm.OLS(lb3Nfl, predict3Nfl)
res3Nfl = mdl3Nfl.fit()
mdl1Nfl = sm.OLS(lb1Nfl, predict1Nfl)
res1Nfl = mdl1Nfl.fit()
mdl2Nfl = sm.OLS(lb2Nfl, predict2Nfl)
res2Nfl = mdl2Nfl.fit()

tstErr = [0 for i in range(10)] 
for i in range(len(test_files)):
    if(i==0):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)
        predictors = sm.add_constant(predictors)        
        testLbPredict1 = res1SB.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict1 - labels))
    if(i==1):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)
        predictors = sm.add_constant(predictors)        
        testLbPredict2 = res2SB.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict2 - labels))      
    if(i==2):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)        
        testLbPredict3 = res3SB.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict3 - labels)) 
    if(i==3):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)        
        testLbPredict4 = res1Nfl.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict4 - labels))         
    if(i==4):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)        
        testLbPredict5 = res1Nfl.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict5 - labels))         
    if(i==5):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)        
        testLbPredict6 = res2SB.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict6 - labels))         
    if(i==6):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)              
        testLbPredict7 = res3Nfl.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict7 - labels))         
    if(i==7):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = np.insert(predictors, [0], [[1],[1],[1],[1],[1]], axis=1)     
        testLbPredict8 = res2SB.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict8 - labels))     
    if(i==8):
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = sm.add_constant(predictors)        
        testLbPredict9 = res2SB.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict9 - labels)) 
    if(i==9):
        i=9
        file_name = os.path.join("test_data", test_files[i])
        modHrFeatures = getHrlyFeat(file_name)    
        predictors, labels = getLabelsMatrix(modHrFeatures)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = np.insert(predictors, [0], [[1],[1],[1],[1],[1],[1]], axis=1)     
        testLbPredict10 = res3Nfl.predict(predictors)
        tstErr[i] = np.mean(abs(testLbPredict10 - labels)) 