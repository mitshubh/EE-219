# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:48:34 2017

@author: swati.arora
"""

## alpha values      rmse values
#    1               3.85904839402 
#    0.1             3.87963373235
#    0.01            3.88680112828
#    0.001           3.88763036765




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn import cross_validation 

housing = pd.read_csv("housing_data.csv")

#housing_X = housing[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
#housing_Y = housing[['MEDV']]

housing_X = housing.iloc[:,0:13]
housing_Y = housing.iloc[:,13].reshape(len(housing),1)

total_error = 0
rmse = range(4)
alpha_list = [1, 0.1,0.01,0.001]

for counter in range(4):
    total_error = 0
    regression = lm.Ridge(alpha =alpha_list[counter])
    for i in range(10):
        housing_X_train,housing_X_test, housing_Y_train, housing_Y_test =  cross_validation.train_test_split(housing_X, housing_Y, test_size=0.1, random_state=42)
        regression.fit(housing_X_train,housing_Y_train)
    
        housing_predicted = regression.predict(housing_X_test)
    
        error = housing_predicted - housing_Y_test
        dotProdError = np.dot(np.transpose(error),error)
        total_error += dotProdError[0][0]

    rmse_10fold = np.sqrt(total_error/len(housing_X))
    rmse[counter] = rmse_10fold

plt.figure()
plt.xlabel("ALPHA")
plt.ylabel("RMSE value")
plt.title("Ridge Regression")
plt.plot(alpha_list,rmse, c="blue")