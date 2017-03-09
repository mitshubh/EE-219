# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:21:01 2017

@author: amehrotra
"""


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, cluster
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import numpy as np
import string
import matplotlib.pyplot as plt
import pandas as pd
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
total_labels = total.target//4; 


total_tfidf_n = normalize(total_tfidf,norm='l2',axis=1,copy=True)
nmf = NMF(n_components = 5,init = 'random', random_state=0)
tfidf_matrix_r1 = nmf.fit_transform(total_tfidf_n)
pca = PCA(n_components=2)
tfidf_matrix_r2 = pca.fit_transform(tfidf_matrix_r1)
tfidf_matrix_r = np.log(np.add(tfidf_matrix_r2,1.0))

km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, precompute_distances='auto', verbose=0,
            copy_x=True, n_jobs=1).fit(tfidf_matrix_r)
            
conf_mat = confusion_matrix(total_labels, km.labels_)
print(conf_mat)        

count = 0
for l in km.labels_:
    if(total_labels[count] == 0 and l == 0):
        total_labels[count] =0
    elif(total_labels[count] == 0 and l == 1):
        total_labels[count] =1
    elif(total_labels[count] == 1 and l == 1):
        total_labels[count] = 2
    elif(total_labels[count] == 1 and l == 0):
       total_labels[count] = 3
    count = count+1
    
x,y = tfidf_matrix_r[:,0], tfidf_matrix_r[:,1]
x1,y1 = x, y
df = pd.DataFrame(dict(x=x1, y=y1, label=total_labels))

groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.x, group.y, marker='.', linestyle='', label=name)
ax.legend()







    
