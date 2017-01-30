# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 21:11:29 2017

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############################################################################################
#Task 1
data = pd.read_csv("C:\\Users\\Admin\\Google Drive\Future\\UCLA\\Winter 2017\\EE-219\\Project1\\network_backup_dataset.csv")
daysOfTheWeek = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
data['Day of Week'] = data['Day of Week'].apply(lambda x:daysOfTheWeek[x])
data['Day no'] = (data['Week #']-1)*7 +  data['Day of Week']
new_data= data[data['Day no'].map(lambda i : 1 <= i and i <= 20)]
new_data = new_data[['Work-Flow-ID','Day no', 'Size of Backup (GB)']]
new_data.groupby(['Work-Flow-ID','Day no']).sum().plot(y='Size of Backup (GB)')
plt.show()












