# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:43:43 2017

@author: swati.arora
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
from sklearn.preprocessing import normalize
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

print("printing dimensions for tfidf \n")
print(total_tfidf.shape)

print "Clustering for all categories \n"

# Analyzing SVD
print "Performing Analysis for SVD \n"
dims = 21;
for comp in np.arange(2,dims):
    svd_model = TruncatedSVD(n_components=comp, random_state=42)    # SVD model, k=comp
    lsi_1 = svd_model.fit_transform(total_tfidf)  #Apply LSI
    lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
    kmeans = KMeans(n_clusters=20).fit(lsi)
    conf_mat = confusion_matrix(total_labels, kmeans.labels_)  #Confusion Matrix
    kmeansLab = kmeans.labels_
    print("For dimension  : ", comp)    
    print("homogeneity score : ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
    print("adjusted rand score : ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 


best_comp = 17      # From analysis above
svd_model = TruncatedSVD(n_components=best_comp, random_state=42)    # SVD model, k=comp
lsi_1 = svd_model.fit_transform(total_tfidf)  #Apply LSI
lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
kmeans = KMeans(n_clusters=20).fit(lsi)
conf_mat_svd = confusion_matrix(total_labels, kmeans.labels_)  #Confusion Matrix
kmeansLab = kmeans.labels_
print "best results at n_components = ", best_comp   
print("Confusion Matrix :  \n", conf_mat_svd)    
# Results
print("homogeneity score : ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
print("completeness score :  ", cluster.completeness_score(total_labels, kmeans.labels_)) # completeness score
print("adjusted rand score : ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 
print("adusted mutual info score : ", cluster.adjusted_mutual_info_score(total_labels, kmeans.labels_)) # adusted mutual info score
 

   
# Analyzing NMF ------------------------------------
print("Performing Analysis for NMF\n")
dims = 21;
for comp in np.arange(2,dims):
    nmf_model = NMF(n_components=comp, init='random', random_state=0)    # NMF model, k=comp
    lsi_1 = nmf_model.fit_transform(total_tfidf)  #Apply LSI
    lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
    kmeans = KMeans(n_clusters=20).fit(lsi)
    conf_mat = confusion_matrix(total_labels, kmeans.labels_);  #Confusion Matrix
    print("For dimension : ", comp)    
    #Results
    print("homogeneity score : ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
    print("adjusted rand score : ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 


best_comp = 18      # From analysis above
nmf_model = NMF(n_components=best_comp, init='random', random_state=0)    # NMF model, k=comp
lsi_1 = nmf_model.fit_transform(total_tfidf)  #Apply LSI
lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
kmeans = KMeans(n_clusters=20).fit(lsi)
conf_mat_nmf = confusion_matrix(total_labels, kmeans.labels_)  #Confusion Matrix
kmeansLab = kmeans.labels_
print "best results at n_components = ", best_comp   
print("Confusion Matrix :  \n", conf_mat_nmf)    
# Results
print("homogeneity score : ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
print("completeness score :  ", cluster.completeness_score(total_labels, kmeans.labels_)) # completeness score
print("adjusted rand score : ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 
print("adusted mutual info score : ", cluster.adjusted_mutual_info_score(total_labels, kmeans.labels_)) # adusted mutual info score
 


homogeneity = []
completeness = []
p = np.arange(10, 100)
for k in np.arange(10, 100):
    svd_model = TruncatedSVD(n_components=17, random_state=42)    # SVD model, k=comp
    lsi_1 = svd_model.fit_transform(total_tfidf)                #Apply LSI
    lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
    kmeans = KMeans(n_clusters=k).fit(lsi)
    conf_mat = confusion_matrix(total_labels, kmeans.labels_)   #Confusion Matrix
    homogeneity.append(cluster.homogeneity_score(total_labels, kmeans.labels_))
    completeness.append(cluster.completeness_score(total_labels, kmeans.labels_))


    
homo_array = np.asarray(homogeneity)
comp_array = np.asarray(completeness)
plt.plot(p,completeness, color='navy', lw=1, linestyle='--')
plt.plot(p,homogeneity,color='orange', lw=1, linestyle='--')
idx = np.argwhere(np.diff(np.sign(comp_array - homo_array)) != 0).reshape(-1) + 0
plt.plot(p[idx], comp_array[idx], 'ro')
plt.show()

d = comp_array - homo_array
for i in range(len(d) - 1):
    if d[i] == 0. or d[i] * d[i + 1] < 0.:
        print p[i] 
        
homogeneity = []
completeness = []
p = np.arange(20, 30)        
for k in np.arange(20, 30):
    svd_model = TruncatedSVD(n_components=17, random_state=42)    # SVD model, k=comp
    lsi_1 = svd_model.fit_transform(total_tfidf)                #Apply LSI
    lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
    kmeans = KMeans(n_clusters=k).fit(lsi)
    conf_mat = confusion_matrix(total_labels, kmeans.labels_)   #Confusion Matrix
    homogeneity.append(cluster.homogeneity_score(total_labels, kmeans.labels_))
    completeness.append(cluster.completeness_score(total_labels, kmeans.labels_))


homo_array = np.asarray(homogeneity)
comp_array = np.asarray(completeness)
plt.plot(p,completeness, color='navy', lw=1, linestyle='--')
plt.plot(p,homogeneity,color='orange', lw=1, linestyle='--')
idx = np.argwhere(np.diff(np.sign(comp_array - homo_array)) != 0).reshape(-1) + 0
plt.plot(p[idx], comp_array[idx], 'ro')
plt.show()

d = comp_array - homo_array
for i in range(len(d) - 1):
    if d[i] == 0. or d[i] * d[i + 1] < 0.:
        print p[i] 
    


print "\n" 
print "For best value of K = 22, scores are :"
svd_model = TruncatedSVD(n_components=17, random_state=42)    # SVD model, k=comp
lsi_1 = svd_model.fit_transform(total_tfidf)                #Apply LSI
lsi = normalize(lsi_1,norm='l2',axis=1,copy=True)
kmeans = KMeans(n_clusters=22).fit(lsi)
conf_mat = confusion_matrix(total_labels, kmeans.labels_)   #Confusion Matrix   
print("Confusion Matrix -- \n", conf_mat)   
#    Results
print("homogeneity score -- ", cluster.homogeneity_score(total_labels, kmeans.labels_)) # homogeneity score
print("completeness score -- ", cluster.completeness_score(total_labels, kmeans.labels_)) # completeness score
print("adjusted rand score -- ", cluster.adjusted_rand_score(total_labels, kmeans.labels_)) # adjusted rand score 
print("adusted mutual info score -- ", cluster.adjusted_mutual_info_score(total_labels, kmeans.labels_)) # adusted mutual info score

    
 
