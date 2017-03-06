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
total_tfidf = tfidf_transformer.fit_transform(total_counts)
#Normalize tfidf results ? using TfIdfVectorizer

total_labels = total.target//4; # Integer Division

print("printing dimensions for tfidf")
print(total_tfidf.shape)

# Checking if some permutation on rows makes the matrix almost diagonal
# perm = np.arange(total_counts.shape[0])
# permutated_tfidf = total_tfidf[perm,:]
# permutated_labels = total_labels[perm]

print("Scipy")
U, s, V = scipy.sparse.linalg.svds(total_tfidf)
print(U.shape)
print(s.shape)
print(V.shape)
print(s)
dimension, = s.shape



print("Performing Analysis for SVD\n")
# Singular Value Decomposition -----------------------
# Explained Variance Sums -- 50 - 10%, 100 - 15%, 500 - 39%, 1000 - 55%
compArr = [2,10,50,100,500]
for comp in compArr:
    svd_model = TruncatedSVD(n_components=comp)
    lsi = svd_model.fit(total_tfidf)
    explained_variance = svd_model.explained_variance_ratio_.sum()
    print("\nExplained variance of the SVD step with {} components: {}%".format(comp, int(explained_variance * 100)))

#Since we're interested in capturing at least 25-35% variance, we pick number of components as 100
dims = 100;
result = 2;
hom_result = 0;
conf = [] ;
for comp in np.arange(2,dims):
    svd_model = TruncatedSVD(n_components=comp, random_state=42)    # SVD model, k=comp
    lsi = svd_model.fit_transform(total_tfidf)  #Apply LSI
    kmeans = KMeans(n_clusters=2).fit(lsi)
    conf_mat = confusion_matrix(total_labels, kmeans.labels_)  #Confusion Matrix
    score = cluster.homogeneity_score(total_labels, kmeans.labels_)
    #print("For dimension: ", comp)    
    #print("Confusion Matrix -- \n", conf_mat)
    
    if(score > hom_result):
        hom_result = score
        result = comp
        conf = conf_mat
    
    #Results
    #print("homogeneity score -- ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
    #print("completeness score -- ", cluster.completeness_score(total_labels, kmeans.labels_)) # completeness score
    #print("adjusted rand score -- ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 
    #print("adusted mutual info score -- ", cluster.adjusted_mutual_info_score(total_labels, kmeans.labels_)) # adusted mutual info score
#hom_result = 0.80222654947895511
#resut = 65
#conf = 3.791000000000000000e+03	1.120000000000000000e+02
#1.300000000000000000e+02	3.849000000000000000e+03

    
## Analyzing NMF ------------------------------------
#compArr = [2,10,50,100,500]
#for comp in compArr:
#    nmf_model = NMF(n_components=comp, init='random', random_state=0)    # NMF model, k=comp
#    lsi = nmf_model.fit(total_tfidf.T)
#    print("\nReconstrction error with {} components: {}%".format(comp, nmf_model.reconstruction_err_))


#dims = 100;
#for comp in np.arange(2,dims):
#    nmf_model = NMF(n_components=2, init='random', random_state=0)    # NMF model, k=comp
#    lsi = svd_model.fit_transform(total_tfidf)  #Apply LSI
#    kmeans = KMeans(n_clusters=2).fit(lsi)
#    conf_mat = confusion_matrix(total_labels, kmeans.labels_);  #Confusion Matrix
#    print("For dimension: ", comp)    
#    print("Confusion Matrix -- \n", conf_mat)
#    #Results
#    print("homogeneity score -- ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
#    print("completeness score -- ", cluster.completeness_score(total_labels, kmeans.labels_)) # completeness score
#    print("adjusted rand score -- ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 
#    print("adusted mutual info score -- ", cluster.adjusted_mutual_info_score(total_labels, kmeans.labels_)) # adusted mutual info score
