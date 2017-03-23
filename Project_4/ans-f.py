# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:37:06 2017

@author: swati.arora
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, cluster
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import string
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

def stemList(stringList):
    stemmer = SnowballStemmer("english")
    data=[]
    punctuations = list(string.punctuation)
    punctuations.append("''")
    # Stem and Remove Punctuations in one go
    for s in stringList:
        data.append(' '.join([stemmer.stem(i.strip("".join(punctuations))) for i in word_tokenize(s) if i not in punctuations]))
    return data
    
categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

total = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

cVectorizer = text.CountVectorizer(min_df=1, stop_words = text.ENGLISH_STOP_WORDS)
total_counts = cVectorizer.fit_transform(stemList(total.data))

tfidf_transformer = text.TfidfTransformer()
total_tfidf = tfidf_transformer.fit_transform(total_counts)

total_labels = total.target

print("printing dimensions for tfidf")
print(total_tfidf.shape)

clustered_labels = list(total_labels)
i = 0
for i in range(len(total_labels)):
    if total_labels[i] == 0 or total_labels[i] == 19 or total_labels[i] == 15:
        clustered_labels[i] = 1    
    elif total_labels[i] == 1 or total_labels[i] == 2 or total_labels[i] == 3  or total_labels[i] == 4 or total_labels[i] == 5:
        clustered_labels[i] = 2    
    elif total_labels[i] == 6:
        clustered_labels[i] = 5    
    elif total_labels[i] == 7 or total_labels[i] == 8 or total_labels[i] == 9 or total_labels[i] == 10:
        clustered_labels[i] = 0     
    elif total_labels[i] == 11 or total_labels[i] == 12 or total_labels[i] == 13 or total_labels[i] == 14: 
        clustered_labels[i] = 4    
    elif total_labels[i] == 16 or total_labels[i] == 17 or total_labels[i] == 18:
        clustered_labels[i] = 3    
    i+= 1

homogeneity = []
completeness = []
p = np.arange(2, 20)
for comp in range(2,20):
    print "Truncating features with SVD with number of componenets as : ",comp
    lsi_1 = normalize(total_tfidf,norm='l2',axis=1,copy=True)
    svd_model = TruncatedSVD(n_components=comp, random_state=42)    # SVD model, k=comp
    lsi = svd_model.fit_transform(lsi_1)  #Apply LSI
#    lsi = normalize(lsi,norm='l2',axis=1,copy=True)
    
    print "K-Means clustering with number of clusters : 6 "
    kmeans = KMeans(n_clusters=6).fit(lsi)
    kmeans_labels = kmeans.labels_
    
    homogeneity.append(cluster.homogeneity_score(clustered_labels, kmeans_labels))
    completeness.append(cluster.completeness_score(clustered_labels, kmeans_labels))
    
    
homo_array = np.asarray(homogeneity)
comp_array = np.asarray(completeness)
plt.plot(p,completeness, color='navy', lw=1, linestyle='--')
plt.plot(p,homogeneity,color='orange', lw=1, linestyle='--')
idx = np.argwhere(np.diff(np.sign(comp_array - homo_array)) != 0).reshape(-1) + 0
plt.plot(p[idx], comp_array[idx], 'ro')
plt.xlabel('Components')
plt.ylabel('Homogenity and Completeness Score')
plt.title('Best componenet for SVD')
plt.legend(loc="lower right")

plt.show()

d = comp_array - homo_array
for i in range(len(d) - 1):
    if d[i] == 0. or d[i] * d[i + 1] < 0.:
        print p[i] 
        
print "\n" 
print "For best value of componenents = 7, scores are :"
lsi_1 = normalize(total_tfidf, norm='l2',axis=1,copy=True)
svd_model = TruncatedSVD(n_components=7, random_state=42)    # SVD model, k=comp
lsi = svd_model.fit_transform(lsi_1)  #Apply LSI
kmeans = KMeans(n_clusters=6).fit(lsi)
kmeans_labels = kmeans.labels_
conf_mat = confusion_matrix(clustered_labels, kmeans_labels)   #Confusion Matrix   
print("Confusion Matrix -- \n", conf_mat)   
#Results
print("homogeneity score -- ", cluster.homogeneity_score(clustered_labels, kmeans_labels)) # homogeneity score
print("completeness score -- ", cluster.completeness_score(clustered_labels, kmeans_labels)) # completeness score
print("adjusted rand score -- ", cluster.adjusted_rand_score(clustered_labels, kmeans_labels)) # adjusted rand score 
print("adusted mutual info score -- ", cluster.adjusted_mutual_info_score(clustered_labels, kmeans_labels)) # adusted mutual info score

    