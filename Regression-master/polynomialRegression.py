# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:20:35 2017

@author: swati.arora
"""

# This program checks till degree 9 for polynomial Refression
# Hence take time to run

# Best degree is 2

## degree values   rmse values
#   1              3.88772390585
#   2              3.254117836
#   3              998.553372066
#   4              118.524650282
#   5              75.3317118668
#   6              88.3090192446
#   7              155.253676941
#   8              229.934159917
#   9              289.458455608

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn import cross_validation 
from sklearn import preprocessing 


housing = pd.read_csv("housing_data.csv")

## housing_X = housing[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
## housing_Y = housing[['MEDV']]

housing_X = housing.iloc[:,0:13]
housing_Y = housing.iloc[:,13].reshape(len(housing),1)

regression = lm.LinearRegression()
total_error = 0
rmse = range(9)
degree_list = range(1,10)

for deg in degree_list:
    ## print " Checking for degree",deg
    total_error = 0 
    for i in range(10):
        housing_X_train,housing_X_test, housing_Y_train, housing_Y_test =  cross_validation.train_test_split(housing_X, housing_Y, test_size=0.1, random_state=42)
        poly = preprocessing.PolynomialFeatures(degree=deg)
        housing_X_train_poly = poly.fit_transform(housing_X_train)
        housing_X_test_poly = poly.fit_transform(housing_X_test)
    
        regression.fit(housing_X_train_poly,housing_Y_train)
    
        housing_predicted = regression.predict(housing_X_test_poly)
    
        error = housing_predicted - housing_Y_test
        dotProdError = np.dot(np.transpose(error),error)
        total_error += dotProdError[0][0]

    rmse_10fold = np.sqrt(total_error/len(housing_X))
    rmse[deg-1] = rmse_10fold

plt.figure()
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.title("Polynomial Regression")
plt.plot(degree_list,rmse, c="blue")