# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 10:25:43 2017

@author: Shubham
"""

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import matplotlib.pyplot as py

categories=['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
#graphics_all = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
graphics_all = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

#Convert the dataset into a pandas dataframe
data_df = pd.DataFrame(data=graphics_all.data,columns=['DocCount'])
target_df = pd.DataFrame(data=graphics_all.target.astype('O'), columns=['types'])

target_df.replace(0,'alt.atheism',inplace=True)
target_df.replace(1,'com.graphics',inplace=True)
target_df.replace(2,'comp.os.ms-windows.misc',inplace=True)
target_df.replace(3,'comp.sys.ibm.pc.hardware',inplace=True)
target_df.replace(4,'comp.sys.mac.hardware',inplace=True)
target_df.replace(5,'comp.windows.x',inplace=True)
target_df.replace(6,'misc.forsale',inplace=True)
target_df.replace(7,'rec.autos',inplace=True)
target_df.replace(8,'rec.motorcycles',inplace=True)
target_df.replace(9,'rec.sport.baseball',inplace=True)
target_df.replace(10,'rec.sport.hockey',inplace=True)
target_df.replace(11,'sci.crypt',inplace=True)
target_df.replace(12,'sci.electronics',inplace=True)
target_df.replace(13,'sci.med',inplace=True)
target_df.replace(14,'sci.space',inplace=True)
target_df.replace(15,'soc.religion.christian',inplace=True)
target_df.replace(16,'talk.politics.guns',inplace=True)
target_df.replace(17,'talk.politics.mideast',inplace=True)
target_df.replace(18,'talk.politics.misc',inplace=True)
target_df.replace(19,'talk.religion.misc',inplace=True)

#target_df.replace(0,'comp.graphics',inplace=True)
#target_df.replace(1,'comp.os.ms-windows.misc',inplace=True)
#target_df.replace(2,'comp.sys.ibm.pc.hardware',inplace=True)
#target_df.replace(3,'comp.sys.mac.hardware',inplace=True)
#target_df.replace(4,'rec.autos',inplace=True)
#target_df.replace(5,'rec.motorcycles',inplace=True)
#target_df.replace(6,'rec.sport.baseball',inplace=True)
#target_df.replace(7,'rec.sport.hockey',inplace=True)

net_df = data_df.join(target_df)

#drawing bar plot for each of the categories
net_count = net_df.groupby('types').count()
net_count = net_count.reset_index('')
net_count.plot(x='types', y='DocCount', kind='bar', legend='false')
py.grid()
py.ylabel('Document #', fontsize=20)
py.title('Plot of document # vs Category', fontsize=25)
py.xticks(rotation=290)
py.draw()
py.show()

#group 8 categories into sets of 2
dict = {'comp.graphics': 'Computer Technology', 'comp.os.ms-windows.misc':'Computer Technology','comp.sys.ibm.pc.hardware':'Computer Technology', 'comp.sys.mac.hardware':'Computer Technology', 'rec.autos':'Recreational activity', 'rec.motorcycles':'Recreational activity', 'rec.sport.baseball':'Recreational activity', 'rec.sport.hockey':'Recreational activity'}
net_count['globalTypes'] = net_count['types'].apply(lambda x: dict[x])
net_count = net_count.groupby('globalTypes').sum()
net_count = net_count.reset_index('')
print(net_count)