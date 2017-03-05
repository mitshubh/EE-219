# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:04:51 2017

@author: Shubham
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import string
from sklearn.decomposition import TruncatedSVD

def stemList(stringList):
    stemmer = SnowballStemmer("english")
    data=[]
    punctuations = list(string.punctuation)
    punctuations.append("''")
    # Stem and Remove Punctuations in one go
    for s in stringList:
        data.append(' '.join([stemmer.stem(i.strip("".join(punctuations))) for i in word_tokenize(s) if i not in punctuations]))
    return data

all_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

tfIdfVector = text.TfidfVectorizer(stop_words = text.ENGLISH_STOP_WORDS)
net_tfidf = tfIdfVector.fit_transform(stemList(all_data.data))

# SVD model, k=50
svd_model = TruncatedSVD(n_components=50, random_state=42)
#Apply LSI
lsi = svd_model.fit_transform(net_tfidf.T)