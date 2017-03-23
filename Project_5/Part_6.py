import re
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import SnowballStemmer
import sklearn.linear_model as sk
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC, SVC
from nltk import word_tokenize
import string



def filterByLoc(dt):
    result = dt[(dt.location.str.contains(r'[.]+ WA$'))
            | (dt.location.str.contains(r'[.]+ MA$'))
            | (dt.location.str.contains(r'[.]+ Washington\s'))
            | (dt.location.str.contains('Seattle'))
            | (dt.location.str.contains('Boston'))
            | (dt.location.str.contains('Massachusetts'))]
    return result
    
#def tokenizeData(data):
def tokenizeData(data):
    stemmer2 = SnowballStemmer("english") # for removing stem words
    stop_words = text.ENGLISH_STOP_WORDS  # omit stop words

    temp = data
    temp = re.sub("[,.-:/()?{}*$#&]"," ",temp)  # remove all symbols
    temp = "".join([ch for ch in temp if ch not in string.punctuation])  # remove all punctuation
    temp = "".join(ch for ch in temp if ord(ch) < 128)  # remove all non-ascii characters
    temp = temp.lower() # convert to lowercase
    words = temp.split()
    no_stop_words = [w for w in words if not w in stop_words]  # stemming of words
    stemmedData = [stemmer2.stem(plural) for plural in no_stop_words]

    return stemmedData

def mapLoc(data):
    targ = []
    for location in data.location.apply(lambda x: x.encode('utf-8').strip()):
        if (r'[.]+ WA$' in location) or ('Seattle' in location) or (r'[.]+ Washington\s' in location):
            targ.append(1)
        else:
            targ.append(0)
    return np.array(targ)

def balDataSets(data, targ):
    nwDt = data.copy()
    if (len(targ[targ==0])) > (len(targ[targ==1])):
        pts = len(targ[targ==0]) - len(targ[targ==1])
        indices = np.where(targ == 1)
    else:
        pts = len(targ[targ==1]) - len(targ[targ==0])
        indices = np.where(targ == 0)

    np.random.shuffle(indices)
    indices = np.resize(indices, pts)
    nwDt = nwDt.append(data.iloc[indices])
    targToAdd = targ[indices]
    nwTarg = np.concatenate([targ, targToAdd])
    return nwDt, nwTarg

DATA_FOLDER = 'tweet_data/'
filename = 'tweets_#superbowl.txt'

tweets_ = []
with open(DATA_FOLDER + filename, 'r') as f:
    for row in f:
        jrow = json.loads(row)
        d = {
            'tweet': jrow['title'],
            'location': jrow['tweet']['user']['location']
        }
        tweets_.append(d)
allDt = pd.DataFrame(tweets_)
redByLoc = filterByLoc(allDt)
allTarg = mapLoc(redByLoc)
data, trainTarget = balDataSets(redByLoc, allTarg)

vectorizer = CountVectorizer(analyzer='word', stop_words='english', tokenizer=tokenizeData)
tfidfTransformer = TfidfTransformer()
trainCount = vectorizer.fit_transform(data.tweet)
trainTfidf = tfidfTransformer.fit_transform(trainCount)

svd = TruncatedSVD(n_components=50, random_state=42)
trainReduced = svd.fit_transform(trainTfidf)
minMaxScale = preprocessing.MinMaxScaler()
trainData = minMaxScale.fit_transform(trainReduced)

k=5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracy = 0
for trainIndex, testIndex in kf.split(trainData):
    xTrain, xTest = trainData[trainIndex], trainData[testIndex]
    yTrain, yTest = trainTarget[trainIndex], trainTarget[testIndex]

    clf = MultinomialNB().fit(xTrain, yTrain)
    predicted_bayes = clf.predict(xTest)
    accBayes = np.mean(predicted_bayes == yTest)
    accuracy += accBayes

print "Average CV-Accuracy of Multinomial Naive Bayes: " + str(accuracy/k)
print(classification_report(yTest, predicted_bayes))
print "Confusion Matrix:"
print(confusion_matrix(yTest, predicted_bayes))

fpr, tpr, thresholds = roc_curve(yTest, predicted_bayes)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multinomial Naive Bayes: Receiver Operating Characteristic Curve')
plt.show()

accuracy = 0
for trainIndex, testIndex in kf.split(trainData):
    xTrain, xTest = trainData[trainIndex], trainData[testIndex]
    yTrain, yTest = trainTarget[trainIndex], trainTarget[testIndex]

    logit = sk.LogisticRegression().fit(xTrain, yTrain)
    probabilities = logit.predict(xTest)
    predictedLr = (probabilities > 0.5).astype(int)
    accLr = np.mean(predictedLr == yTest)
    accuracy += accLr

print "Average CV-Accuracy of Logistic Regression: " + str(accuracy/k)
print(classification_report(yTest, predictedLr))
print "Confusion Matrix:"
print(confusion_matrix(yTest, predictedLr))

fpr, tpr, thresholds = roc_curve(yTest, predictedLr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression: Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

accuracy = 0
for trainIndex, testIndex in kf.split(trainData):
    xTrain, xTest = trainData[trainIndex], trainData[testIndex]
    yTrain, yTest = trainTarget[trainIndex], trainTarget[testIndex]


    linearSVM = LinearSVC(dual=False, random_state=42).fit(xTrain, yTrain)
    predicted_svm = linearSVM.predict(xTest)
    accSVM = np.mean(predicted_svm == yTest)
    accuracy += accSVM

print "Average CV-Accuracy of Linear SVM: " + str(accuracy/k)
print(classification_report(yTest, predicted_svm))
print "Confusion Matrix:"
print(confusion_matrix(yTest, predicted_svm))

fpr, tpr, thresholds = roc_curve(yTest, predicted_svm)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Linear SVM: Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()