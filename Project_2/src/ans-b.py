# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:16:45 2017

@author: Shubham
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import string

def stemList(stringList):
    stemmer = SnowballStemmer("english")
    data=[]
    punctuations = list(string.punctuation)
    punctuations.append("''")
    # Stem and Remove Punctuations in one go
    for s in stringList:
        data.append(' '.join([stemmer.stem(i.strip("".join(punctuations))) for i in word_tokenize(s) if i not in punctuations]))
    return data
    
#Alternate way of stemming
'''
stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

stem_vectorizer = CountVectorizer(analyzer=stemmed_words)
train_counts = cVectorizer.fit_transform(stemList(train.data))
'''

categories=['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

cVectorizer = text.CountVectorizer(min_df=1, stop_words = text.ENGLISH_STOP_WORDS)
train_counts = cVectorizer.fit_transform(stemList(train.data))

tfidf_transformer = text.TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
print(train_tfidf.shape)