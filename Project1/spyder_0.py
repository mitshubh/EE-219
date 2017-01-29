# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:59:20 2017

@author: Shubham
"""

import pandas as pd
import matplotlib.pyplot as pyplot


nw_data = pd.read_csv("C:\\Users\\Admin\\Google Drive\\Future\\UCLA\\Winter 2017\\EE-219\\network_backup_dataset.csv")
#Extract a 20 day summary 
nw_data = nw_data.where((nw_data["Week #"]<3) | ((nw_data["Week #"]==3) & (nw_data["Day of Week"]!="Sunday"))).dropna()

#Extract subsequent 20 day summary
#nw_data = nw_data.where(((nw_data["Week #"]>3) & (nw_data["Week #"]<6)) | ((nw_data["Week #"]==6) & (nw_data["Day of Week"]!="Sunday"))).dropna()

#Create a column w.r.t. days
days = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
nw_data["Day #"] = nw_data["Day of Week"].apply(lambda x: days[x])

#Extract relevant content out of the data
nw_data = nw_data[["Day #", "Work-Flow-ID", "Size of Backup (GB)"]]

fig, ax = pyplot.subplots()
for group, frame in nw_data.groupby(["Work-Flow-ID"]):
	nw_agg = frame.groupby(["Day #"]).sum()
	nw_agg = nw_agg.reset_index()
	#print (nw_agg)
	ax = nw_agg.plot(x="Day #", y="Size of Backup (GB)",  kind='line', ax=ax, label=group)
	
pyplot.legend()
pyplot.ylabel('Size of Backup (GB)')
ax.set_title('Size (GB) vs Day # - First 20 days')
pyplot.show()