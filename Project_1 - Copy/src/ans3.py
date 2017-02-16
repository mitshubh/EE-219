import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import linear_model, preprocessing, metrics, ensemble
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

nw_data = pd.read_csv("./network_backup_dataset.csv")
X = nw_data
#Transform X to contain only numeric values
days = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
X["Day #"] =nw_data["Day of Week"].apply(lambda x: days[x])
X["Work-Flow #"] =nw_data["Work-Flow-ID"].apply(lambda x: int(x[-1]))
X["File #"] =nw_data["File Name"].apply(lambda x: int(x.split("_")[-1]))
#X = X[["Week #", "Day #", "Backup Start Time - Hour of Day", "Work-Flow #", "File #", "Backup Time (hour)"]]
X = X[["Week #", "Day #", "Backup Start Time - Hour of Day", "Work-Flow-ID", "File #", "Backup Time (hour)", "Size of Backup (GB)"]]

#marker = itertools.cycle((',', '+', '.', 'o', '*', '|', '^', 's', 'v', 'x'))
colors = ['g', 'r', 'y', 'b', 'm']
data_original = X
RMSE_Arr=[]
i=0
for wid in pd.unique(X['Work-Flow-ID']):
	lab = wid
	X = data_original
	new_data = X[X['Work-Flow-ID'] == wid]
	Y = new_data['Size of Backup (GB)']
	X = new_data.drop('Size of Backup (GB)',axis = 1)
	X = new_data.drop('Work-Flow-ID',axis = 1)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=50)
	#normalize
	normalized_X = preprocessing.normalize(X_train)
	normalized_X_test = preprocessing.normalize(X_test)

	model = linear_model.LinearRegression()
	
	model.fit(normalized_X, Y_train.values)

	predicted_train = model.predict(normalized_X)
	predicted_test = model.predict(normalized_X_test)
	coef_strength = pd.DataFrame(list(zip(X_train.columns, model.coef_)), columns=["features", "estimatedCoefficients"])
	print(wid)
	print(coef_strength)

	#Scatter plot -- predicted vs observed
	fig1 = plt.figure(1)
	frame1 = fig1.add_axes((.1,.3,.8,.6))
	plt.title(("Piece-wise linear regression by workflow with RMSE = {}").format(np.average(RMSE_Arr)))
	plt.scatter(predicted_test, Y_test, color=colors[i], label=lab)
	plt.plot(predicted_test, predicted_test, 'black')
	RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predicted_test))
	print(("Root mean squared error {}").format(RMSE))
	RMSE_Arr.append(RMSE)
	frame1.set_xticklabels([])
	plt.ylabel("Observed Size of Backup (GB)")	
	plt.grid()
	plt.legend()

	#Residual Plots
	frame2 = fig1.add_axes((.1,.1,.8,.2))
	#plt.scatter(predicted_train, Y_train-predicted_train, c='b', s=40, alpha=0.3)
	plt.scatter(predicted_test, Y_test-predicted_test, color=colors[i], label=lab)
	plt.hlines(y=0, xmin=0, xmax=1)
	plt.ylabel("Residual")
	plt.grid()
	i=i+1
	#plt.show()

numpy_arr = np.array(RMSE_Arr)
print(("RMSE average is {}").format(np.sqrt(np.mean(numpy_arr**2))))
print(("Cross Validation error is {}").format(np.average(RMSE_Arr)))
plt.xlabel("Fitted Values - Size of Backup (GB)")
#plt.scatter(RMSE_Arr)
plt.show()