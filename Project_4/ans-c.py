# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:00:21 2017

@author: Shubham
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, cluster
from sklearn.decomposition import TruncatedSVD, NMF
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import numpy as np
import string
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def stemList(stringList):
    stemmer = SnowballStemmer("english")
    data=[]
    punctuations = list(string.punctuation)
    punctuations.append("''")
    # Stem and Remove Punctuations in one go
    for s in stringList:
        data.append(' '.join([stemmer.stem(i.strip("".join(punctuations))) for i in word_tokenize(s) if i not in punctuations]))
    return data

categories=['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
total = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

cVectorizer = text.CountVectorizer(min_df=1, stop_words = text.ENGLISH_STOP_WORDS)
total_counts = cVectorizer.fit_transform(stemList(total.data))

tfidf_transformer = text.TfidfTransformer()
total_tfidf = tfidf_transformer.fit_transform(total_counts)

total_labels = total.target//4 # Integer Division

print("printing dimensions for tfidf")
print(total_tfidf.shape)
print("Performing Analysis for SVD\n")
dims = 20;

for comp in np.arange(2,dims):
    svd_model = TruncatedSVD(n_components=comp, random_state=42)    # SVD model, k=comp
    lsi_1 = svd_model.fit_transform(total_tfidf)  #Apply LSI
    lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
    kmeans = KMeans(n_clusters=2).fit(lsi)
    conf_mat = confusion_matrix(total_labels, kmeans.labels_)  #Confusion Matrix  
    print("For dimension: ", comp)    
    print("Confusion Matrix -- \n", conf_mat)    
    #Results
    print("homogeneity score -- ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
    print("completeness score -- ", cluster.completeness_score(total_labels, kmeans.labels_)) # completeness score
    print("adjusted rand score -- ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 
    print("adusted mutual info score -- ", cluster.adjusted_mutual_info_score(total_labels, kmeans.labels_)) # adusted mutual info score


    
# Analyzing NMF ------------------------------------
print("Performing Analysis for NMF\n")
dims = 20;
for comp in np.arange(2,dims):
    nmf_model = NMF(n_components=comp, init='random', random_state=0)    # NMF model, k=comp
    lsi_1 = nmf_model.fit_transform(total_tfidf)  #Apply LSI
    lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
    kmeans = KMeans(n_clusters=2).fit(lsi)
    conf_mat = confusion_matrix(total_labels, kmeans.labels_);  #Confusion Matrix
    print("For dimension: ", comp)    
    print("Confusion Matrix -- \n", conf_mat)
    #Results
    print("homogeneity score -- ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
    print("completeness score -- ", cluster.completeness_score(total_labels, kmeans.labels_)) # completeness score
    print("adjusted rand score -- ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 
    print("adusted mutual info score -- ", cluster.adjusted_mutual_info_score(total_labels, kmeans.labels_)) # adusted mutual info score
