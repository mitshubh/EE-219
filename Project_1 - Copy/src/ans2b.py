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
#X = X[["Week #", "Day #", "Backup Start Time - Hour of Day", "Work-Flow #", "File #", "Backup Time (hour)"]]
#X = X[["Day #", "Backup Start Time - Hour of Day", "Work-Flow #", "Backup Time (hour)"]]

X = X[["Week #", "Day of Week", "Backup Start Time - Hour of Day", "Work-Flow #", "Backup Time (hour)"]]
categorical_columns = ["Day of Week"]
X = pd.get_dummies(data=X, columns=categorical_columns)


#Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=50)

#normalize
normalized_X = preprocessing.normalize(X_train)
normalized_X_test = preprocessing.normalize(X_test)

#Tune parameters -- number of trees

RMSE=0
RMSE_new=0
i=0
layer=[]
RMSE_arr=[]
iter_arr = np.arange(1, 50)

for i in np.arange(1, 50):
	rfregressor = ensemble.RandomForestRegressor(n_estimators=40, max_depth=i)
	rfregressor.fit(normalized_X, Y_train.values)
	predicted_test = rfregressor.predict(normalized_X_test)
	RMSE = metrics.mean_squared_error(Y_test, predicted_test)
	print(RMSE)
	RMSE_arr.append(RMSE)
	i=i+1
	
RMSE_min = np.min(RMSE_arr)
iter_min = np.argmin(RMSE_arr)
#Plot RMSE vs #Nodes for a single layer
plt.scatter(iter_arr, RMSE_arr)
plt.scatter(iter_min, RMSE_min, color = 'g')
plt.xlabel("No. of Trees", fontsize=16)
plt.ylabel("RMSE", fontsize=16)
plt.title(("RMSE vs Trees # -- min RMSE = {} at {} trees and {} node depth").format(RMSE_min, 40, iter_min), fontsize=20)
plt.grid()
plt.show()

#continue normal plotting

RMSE=0
RMSE_new=0
i=0
#Fit model using Random Forest Regressor -- tune parameters -- changing deceision tree count has no effect

#rfregressor = ensemble.RandomForestRegressor(n_estimators=20+i, max_depth=10)
#rfregressor.fit(normalized_X, Y_train.values)
#predicted_test = rfregressor.predict(normalized_X_test)
#RMSE_new = metrics.mean_squared_error(Y_test, predicted_test)
print(("Root Mean Squared Error: ({}), iter: ({})\n").format(RMSE_min, i))
#RMSE = RMSE_new
print(rfregressor.feature_importances_)
print("\nFeature vs coefficient strength: \n")
coef_strength = pd.DataFrame(list(zip(X_train.columns, rfregressor.feature_importances_)), columns=["features", "estimatedCoefficients"])
print(coef_strength)

print("Root Mean Squared Error: %.5f\n" % RMSE)		
#Scatter plot -- predicted vs observed
fig1 = plt.figure(1)
frame1 = fig1.add_axes((.1,.3,.8,.6))
#plt.scatter(X_test, Y_test)
plt.plot(predicted_test, Y_test, '.b')
plt.plot(predicted_test, predicted_test, '-r')
plt.ylabel("Observed Size of Backup (GB)")
frame1.set_xticklabels([])
plt.grid()
plt.title(("Results after fitting a Random Forests Regressor, RMSE = {}").format(RMSE_min))
#plt.show()

#Residual Plots
frame2 = fig1.add_axes((.1,.1,.8,.2))
#plt.scatter(predicted_train, Y_train-predicted_train, c='b', s=40, alpha=0.3)
plt.scatter(predicted_test, Y_test-predicted_test)
plt.hlines(y=0, xmin=0, xmax=1)
plt.ylabel("Residual")
plt.xlabel("Fitted Value - Size of Backup (GB)")
plt.grid()
plt.show()