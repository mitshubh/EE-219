# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:14:54 2017

@author: swati.arora
"""

# RMSE after 10 fold cross validation: 3.888

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn import cross_validation  

housing = pd.read_csv("housing_data.csv")

## housing_X = housing[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]

housing_X = housing.iloc[:,0:13]
housing_Y = housing.iloc[:,13].reshape(len(housing),1)

regression = lm.LinearRegression()
total_error = 0
for i in range(10):
    housing_X_train,housing_X_test, housing_Y_train, housing_Y_test =  cross_validation.train_test_split(housing_X, housing_Y, test_size=0.1, random_state=42)
    regression.fit(housing_X_train,housing_Y_train)
    housing_predicted = regression.predict(housing_X_test)

    error = housing_predicted - housing_Y_test
    total_error += np.dot(np.transpose(error),error)
    
    plt.figure()
    plt.xlabel("Actual Values")
    plt.ylabel("Fitted values")
    plt.title("Linear Regression")
    plt.scatter(housing_predicted, housing_Y_test, c="blue")
    plt.plot(housing_predicted,housing_predicted, c="red")

rmse_10fold = np.sqrt(total_error/len(housing_X))