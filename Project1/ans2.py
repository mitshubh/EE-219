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

# Hot-bit encoding
#X = X[["Backup Start Time - Hour of Day", "Work-Flow-ID", "Backup Time (hour)"]]
#categorical_columns = ["Work-Flow-ID"]
#X = pd.get_dummies(data=X, columns=categorical_columns)

marker = itertools.cycle((',', '+', '.', 'o', '*', '|', '^', 's', 'v', 'x'))
colors = cmap.rainbow(np.linspace(0, 1, 10))
model = linear_model.LinearRegression()

#Cross-validate
kf = KFold(n_splits=10, shuffle=True)
i=0
RMSE_Arr=[]
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=50)

for train_index, test_index in kf.split(nw_data):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
	print(X_train.shape)
	
	#normalize
	normalized_X = preprocessing.normalize(X_train)
	normalized_X_test = preprocessing.normalize(X_test)

	#Fit Model
	lab="Fold"+str(i+1)
	model.fit(normalized_X, Y_train.values)
	#model.fit(X_train.values, Y_train.values)
	#print("Coefficients: ", model.coef_)
	#print("\nIntercept: ", model.intercept_)
	#Feature related to coefficient strength
	print("\nFeature vs coefficient strength: \n")
	coef_strength = pd.DataFrame(list(zip(X_train.columns, model.coef_)), columns=["features", "estimatedCoefficients"])
	print(coef_strength)
	predicted_train = model.predict(normalized_X)
	predicted_test = model.predict(normalized_X_test)	

	#Root Mean Squared Error
	print("Root Mean Squared Error: %.5f\n" % metrics.mean_squared_error(Y_test, predicted_test))

	#Scatter plot -- predicted vs observed
	fig1 = plt.figure(1)
	frame1 = fig1.add_axes((.1,.3,.8,.6))
	#plt.scatter(X_test, Y_test)
	marker_val = next(marker)
	#plt.title("Plots with 'Backup Start Time - Hour of Day', 'Work-Flow #', 'Backup Time (hour)' as the feature vector")
	plt.scatter(predicted_test, Y_test, color=colors[i], marker=marker_val, label=lab)
	plt.plot(predicted_test, predicted_test, 'black')
	RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predicted_test))
	print(("Root mean squared error {}").format(RMSE))
	RMSE_Arr.append(RMSE)
	#plt.plot(np.arange(0,1,0.1), np.arange(0,1,0.1), '-r')
	frame1.set_xticklabels([])
	plt.ylabel("Observed Size of Backup (GB)")	
	plt.grid()
	plt.legend()
	#plt.show()

	#Residual Plots
	frame2 = fig1.add_axes((.1,.1,.8,.2))
	#plt.scatter(predicted_train, Y_train-predicted_train, c='b', s=40, alpha=0.3)
	plt.scatter(predicted_test, Y_test-predicted_test, color=colors[i], marker=marker_val, label=lab)
	plt.hlines(y=0, xmin=0, xmax=1)
	plt.ylabel("Residual")
	plt.grid()
	i=i+1
	#plt.show()

numpy_arr = np.array(RMSE_Arr)
print(("RMSE average is {}").format(np.sqrt(np.mean(numpy_arr**2))))
print(("Cross Validation error is {}").format(np.average(RMSE_Arr)))
plt.xlabel("Fitted Values - Size of Backup (GB)")
plt.show()