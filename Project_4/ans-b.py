# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:00:42 2017

@author: Shubham
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, cluster
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import numpy as np
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

categories=['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
total = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

cVectorizer = text.CountVectorizer(min_df=1, stop_words = text.ENGLISH_STOP_WORDS)
#train_counts = cVectorizer.fit_transform(stemList(train.data))
total_counts = cVectorizer.fit_transform(stemList(total.data))

tfidf_transformer = text.TfidfTransformer()
#train_tfidf = tfidf_transformer.fit_transform(train_counts)
#print(train_tfidf.shape)
total_tfidf = tfidf_transformer.fit_transform(total_counts)
total_labels = total.target//4; # Integer Division
print(total_tfidf.shape)

# Checking if some permutation on rows makes the matrix almost diagonal
# perm = np.arange(total_counts.shape[0])
# permutated_tfidf = total_tfidf[perm,:]
# permutated_labels = total_labels[perm]

kmeans = KMeans(n_clusters=2, verbose=0, init='random').fit(total_tfidf)
#kmeans = KMeans(n_clusters=2, verbose=0, init='random').fit(permutated_tfidf)

#Confusion Matrix
conf_mat = confusion_matrix(total_labels, kmeans.labels_);
#conf_mat = confusion_matrix(permutated_labels, kmeans.labels_);
print(conf_mat)
print("homogeneity score -- ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
print("completeness score -- ", cluster.completeness_score(total_labels, kmeans.labels_)) # completeness score
print("adjusted rand score -- ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 
print("adusted mutual info score -- ", cluster.adjusted_mutual_info_score(total_labels, kmeans.labels_)) # adusted mutual info score