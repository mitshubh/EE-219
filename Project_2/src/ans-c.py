# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:51:48 2017

@author: Shubham
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer, EnglishStemmer
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

#categories=['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
all_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

#Create net_data (an alternative to train.data) as a list of concatenated strings len(net_data)=|C|
net_data=[]
for category in all_data.target_names:
    category_data = fetch_20newsgroups(subset='all', categories=[category], shuffle=True, random_state=42)
    category_string = ' '.join(category_data.data)
    net_data.append(category_string)

cVectorizer = text.CountVectorizer(min_df=1, stop_words = text.ENGLISH_STOP_WORDS)
net_counts = cVectorizer.fit_transform(stemList(net_data))

# Now, the tfIdf transformer is actually a tfIcf transformer - since we're modified net_counts
tficf_transformer = text.TfidfTransformer()
net_tficf = tficf_transformer.fit_transform(net_counts)

#get max top 10 values in the list for the specified categories
iter=0
for row in net_tficf.toarray():
    if iter==3 or iter==4 or iter==6 or iter==15: 
        print("Top 10 words for category: " + all_data.target_names[iter])
        index_list = sorted(range(len(row)), key=lambda i: -row[i])[0:10]
        for i in index_list:
            print(str(cVectorizer.get_feature_names()[i]) + ' ' + str(net_counts.toarray()[iter][i]))
        print("\n")
    iter=iter+1