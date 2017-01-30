import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import linear_model, preprocessing, metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import itertools

nw_data = pd.read_csv("./network_backup_dataset.csv")
#Assign target
Y = nw_data["Size of Backup (GB)"]
X = nw_data[["Week #", "Day of Week", "Backup Start Time - Hour of Day", "Work-Flow-ID", "File Name", "Backup Time (hour)"]]

#Transform X to contain only numeric values
days = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
X["Day #"] =nw_data["Day of Week"].apply(lambda x: days[x])
X["Work-Flow #"] =nw_data["Work-Flow-ID"].apply(lambda x: int(x[-1]))
X["File #"] =nw_data["File Name"].apply(lambda x: int(x.split("_")[-1]))

#Taking all parameters as features ---------------
X = X[["Week #", "Day #", "Backup Start Time - Hour of Day", "Work-Flow #", "File #", "Backup Time (hour)"]]

#Taking few parameters as features ---------------
#X = X[["Day #", "Backup Start Time - Hour of Day", "Work-Flow #", "Backup Time (hour)"]]

#marker = itertools.cycle((',', '+', '.', 'o', '*', '|', '^', 's', 'v', 'x'))
colors = ['g', 'r', 'y', 'b', 'm']
RMSE_Arr=[]
i=0

#First plot RMSE vs degree of polynomial
# For RMSE vs Folds plot, keep degree constant

RMSE_Avg_Arr=[]
for degree in np.arange(10,10,1):
	i=0
	#RMSE_Arr=[]
	degree=6
	kf = KFold(n_splits=10, shuffle=True)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
		#print("Degree of polynomial:" + str(degree))
		#normalize
		normalized_X = preprocessing.normalize(X_train)
		normalized_X_test = preprocessing.normalize(X_test)

		#Fit Model
		lab="Fold"+str(i+1)
		poly=preprocessing.PolynomialFeatures(degree=degree)
		train_poly = poly.fit_transform(X_train)
		predict_poly = poly.fit_transform(X_test)
		
		#generate the regression object
		model = linear_model.LinearRegression()
		#preform the actual regression
		model.fit(train_poly, Y_train)
		
		predicted_test = model.predict(predict_poly)	
		
		#Root Mean Squared Error
		#print("Root Mean Squared Error: % metrics.mean_squared_error(Y_test, predicted_test))

		#Scatter plot -- predicted vs observed
		#marker_val = next(marker)
		#plt.title("Plots with 'Backup Start Time - Hour of Day', 'Work-Flow #', 'Backup Time (hour)' as the feature vector")
		#plt.scatter(predicted_test, Y_test, color=colors[i], marker=marker_val, label=lab)
		#plt.plot(predicted_test, predicted_test, 'black')
		RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predicted_test))
		#print(("Root mean squared error {}").format(RMSE))
		RMSE_Arr.append(RMSE)
		#plt.plot(np.arange(0,1,0.1), np.arange(0,1,0.1), '-r')
	avg_RMSE = np.average(np.array(RMSE_Arr))
	print(avg_RMSE)
	RMSE_Avg_Arr.append(avg_RMSE)
	degree=degree+1
#plt.scatter(degree, RMSE_Avg_Arr)
#plt.title("Average RMSE vs degree of polynomial, minimum RMSE = 0.02038 at degree=6", fontsize=20)
#py.ylabel("Average RMSE for 10-fold cross validation")
#plt.annotate('threshold', xy=(6, 0.02038), xytext=(7, 0.05), arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.xlabel("Degree of Polynomial")
#plt.show()

#RMSE Across k-folds
plt.scatter(np.arange(1,10,1), RMSE_Arr)