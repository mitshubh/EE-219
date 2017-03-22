import os
import json
import datetime
from sets import Set
from datetime import timedelta
import os.path
import statsmodels.api as sm
from sklearn.cross_validation import KFold
import numpy as np



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
            temp = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':curr.hour,'avg_user_mention_count':0,
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


def cross_validation(predictors,labels):
    predErr = []
    kf = KFold(len(predictors), n_folds=10)
    for train, test in kf:
        trainPredict = [predictors[i] for i in train]
        testPredict = [predictors[i] for i in test]
        trainLb = [labels[i] for i in train]
        tstLb = [labels[i] for i in test]
        trainLb = sm.add_constant(trainLb)        
        model = sm.OLS(trainLb, trainPredict)
        results = model.fit()
        tstLbPredict = results.predict(testPredict)
        pErr = abs(tstLbPredict - tstLb)
        pErr = np.mean(pErr)
        predErr.append(pErr)
    print(predErr)
    return np.mean(predErr)        
        
folder_name = "tweet_data"
train_files =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

for i in range(len(train_files)):
    modHF, period1, period2, period3 = getPeriodHourlyFeatures(i)
    predictors, labels = getLabelsMatrix(modHF)
    average_cv_pred_error = cross_validation(predictors,labels)
    print "The avg prediction error for full cross-validation of", hashtag_list[i], " is ", average_cv_pred_error
    predictors, labels = getLabelsMatrix(period1)
    average_cv_pred_error = cross_validation(predictors,labels)
    print "The avg prediction error using cross-validation for Period 1 of", hashtag_list[i], " is ", average_cv_pred_error
    predictors, labels = getLabelsMatrix(period2)
    average_cv_pred_error = cross_validation(predictors,labels)
    print "The avg prediction error using cross-validation for Period 2 of", hashtag_list[i], " is ", average_cv_pred_error
    predictors, labels = getLabelsMatrix(period3)
    average_cv_pred_error = cross_validation(predictors,labels)
    print "The avg prediction error using cross-validation for Period 3 of", hashtag_list[i], " is ", average_cv_pred_error
