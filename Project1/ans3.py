import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import linear_model, preprocessing, metrics, ensemble
import matplotlib.pyplot as plt

nw_data = pd.read_csv("./network_backup_dataset.csv")
#Assign target
Y = nw_data["Size of Backup (GB)"]
X = nw_data[["Week #", "Day of Week", "Backup Start Time - Hour of Day", "Work-Flow-ID", "File Name", "Backup Time (hour)"]]

#Transform X to contain only numeric values
days = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
X["Day #"] =nw_data["Day of Week"].apply(lambda x: days[x])
X["Work-Flow #"] =nw_data["Work-Flow-ID"].apply(lambda x: int(x[-1]))
X["File #"] =nw_data["File Name"].apply(lambda x: int(x.split("_")[-1]))
X = X[["Week #", "Day #", "Backup Start Time - Hour of Day", "Work-Flow #", "File #", "Backup Time (hour)"]]
