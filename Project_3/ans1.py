# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 00:52:27 2017

@author: Shubham
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

movies = pd.read_csv("C:\\Users\\Admin\\Google Drive\\Future\\UCLA\\Winter 2017\\EE-219\\Project_3\\ml-latest-small\\movies.csv")
ratings = pd.read_csv("C:\\Users\\Admin\\Google Drive\\Future\\UCLA\\Winter 2017\\EE-219\\Project_3\\ml-latest-small\\ratings.csv")
unique_movieId = movies.movieId.unique()
unique_userId = ratings.userId.unique()
d1 = pd.DataFrame()
# Create a dataframe for each user and append it the global dataframe -- New columns are
# automatically added, with missing data set to NaN
for user in unique_userId:
    df = ratings[ratings.userId==user]
    d1 = d1.append(pd.DataFrame(df['rating'].as_matrix(), columns=[user], index=df['movieId']).transpose())

d1=d1.fillna(0)
R = d1.values
W = R>0
W = W.astype(np.float64, copy=False)

for k in [10, 50, 100]:
    model = NMF(n_components=k, random_state=0)
    U = model.fit_transform(W)
    V = model.components_
    print("Least Squared Error for k = %s is %d" % (k, model.reconstruction_err_))