# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:35:00 2017

@author: Admin
"""

import matplotlib.pyplot as py
import numpy as np
RMSE_arr = [0.01393887902209901, 0.014595410564260933, 0.015316081864567852, 0.015560244131134998, 0.058249052578404729, 0.014509673785732627, 0.014586633532196431, 0.01725814412482932, 0.013843626717499736, 0.016585607498528346]
test = np.arange(1,11,1)
py.scatter(test, RMSE_arr)
py.xlabel("Folds in Cross Validation")
py.ylabel("RMSE")
py.title("RMSE vs k-fold cross validation")
py.grid()
py.show()