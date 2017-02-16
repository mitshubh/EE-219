from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as py
from nltk.stem.snowball import SnowballStemmer
import re
import string
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import sklearn.metrics as smet
from nltk import word_tokenize

def stemData(data):
    stemmer = SnowballStemmer("english") 
    x = data
    x = re.sub("[,.-:/()?{}*$#&]"," ",x)  
    x = "".join([ch for ch in x if ch not in string.punctuation])  
    word = x.lower().split()
    stopWords = text.ENGLISH_STOP_WORDS  
    noStopWords = [w for w in word if not w in stopWords]  
    result = [stemmer.stem(plural) for plural in noStopWords]
    return result

category_list = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
train = fetch_20newsgroups(subset='train',  shuffle=True, random_state=42, categories=category_list)
test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42,  categories=category_list)

for data,pos in zip(train.data,range(len(train.data))):
        tempProcessedData = stemData(data)
        train.data[pos] = ' '.join(tempProcessedData)
for data,pos in zip(test.data,range(len(test.data))):
        tempProcessedData = stemData(data)
        test.data[pos] = ' '.join(tempProcessedData)

stopWords = text.ENGLISH_STOP_WORDS 

countVect = CountVectorizer(stop_words=stopWords, lowercase=True)
tfidfTransformer = TfidfTransformer(norm='l2', sublinear_tf=True)


trainCounts = countVect.fit_transform(train.data)
testCounts = countVect.transform(test.data)
trainIdf = tfidfTransformer.fit_transform(trainCounts)
testIdf = tfidfTransformer.transform(testCounts)

svd = TruncatedSVD(n_components=50)
trainLsi = svd.fit_transform(trainIdf)
testLsi = svd.transform(testIdf)


clfList = [OneVsOneClassifier(GaussianNB()), OneVsOneClassifier(svm.SVC(kernel='linear')), OneVsRestClassifier(GaussianNB()), OneVsRestClassifier(svm.SVC(kernel='linear'))]
clfName = ['Naive Bayes OneVsOneClassifier', 'SVM OneVsOneClassifier ','Naive Bayes OneVsRestClassifier ', 'SVM OneVsRestClassifier ']

for clasif,clasifName in zip(clfList,clfName):
    clasif.fit(trainLsi, train.target)
    predicted = clasif.predict(testLsi)
    accuracy = smet.accuracy_score(test.target,predicted)
    precision = smet.precision_score(test.target, predicted, average='macro')
    recall = smet.recall_score(test.target, predicted, average='macro')
    confusion_matrix = smet.confusion_matrix(test.target,predicted)
    print(clasifName)
    print("Recall:  {0}".format(recall * 100))
    print("Accuracy:  {0}".format(accuracy * 100))
    print("Precision:  {0}".format(precision * 100))
    print("Confusion Matrix:  \n{0}".format(confusion_matrix))





