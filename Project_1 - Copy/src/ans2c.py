import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import linear_model, preprocessing, metrics, ensemble, neural_network
import matplotlib.pyplot as plt

nw_data = pd.read_csv("C:\\Users\Admin\\Google Drive\\Future\\UCLA\\Winter 2017\\EE-219\\network_backup_dataset.csv")
#Assign target
Y = nw_data["Size of Backup (GB)"]
X = nw_data[["Week #", "Day of Week", "Backup Start Time - Hour of Day", "Work-Flow-ID", "File Name", "Backup Time (hour)"]]

#Transform X to contain only numeric values
days = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
X["Day #"] =nw_data["Day of Week"].apply(lambda x: days[x])
X["Work-Flow #"] =nw_data["Work-Flow-ID"].apply(lambda x: int(x[-1]))
X["File #"] =nw_data["File Name"].apply(lambda x: int(x.split("_")[-1]))
#X = X[["Week #", "Day #", "Backup Start Time - Hour of Day", "Work-Flow #", "File #", "Backup Time (hour)"]]

X = X[["Week #", "Day of Week", "Backup Start Time - Hour of Day", "Work-Flow #", "Backup Time (hour)"]]
categorical_columns = ["Day of Week"]
X = pd.get_dummies(data=X, columns=categorical_columns)

#Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=50)

#normalize
normalized_X = preprocessing.normalize(X_train)
normalized_X_test = preprocessing.normalize(X_test)

RMSE=0
RMSE_new=0
i=0
layer=[]
RMSE_arr=[]
iter_arr = np.arange(1, 100)

#Fit model using Nueral Network Regressor
# Tune number of hidden layers
for i in np.arange(1, 100):
	#layer.append(i)
	#print(layer)
	nnreg = neural_network.MLPRegressor(hidden_layer_sizes=(i))
	nnreg.fit(normalized_X, Y_train.values)
	predicted_test = nnreg.predict(normalized_X_test)
	RMSE_new = np.sqrt(metrics.mean_squared_error(Y_test, predicted_test))
	print(("Root Mean Squared Error: ({}), iter: ({})\n").format(RMSE_new, i))
	RMSE_arr.append(RMSE_new)
	#print(nnreg.coefs_)
	#print("\nFeature vs coefficient strength: \n")
	coef_strength = pd.DataFrame(list(zip(X_train.columns, nnreg.coefs_)), columns=["features", "estimatedCoefficients"])
	#print(coef_strength)
	i=i+1
		
RMSE_min = np.min(RMSE_arr)
iter_min = np.argmin(RMSE_arr)
#Plot RMSE vs #Nodes for a single layer
plt.scatter(iter_arr, RMSE_arr)
plt.scatter(iter_min, RMSE_min, color = 'g')
plt.xlabel("No. of Nodes", fontsize=16)
plt.ylabel("RMSE", fontsize=16)
plt.title(("RMSE vs Nodes # -- min RMSE = {} at {} nodes").format(RMSE_min, iter_min), fontsize=20)
plt.grid()
plt.show()

#For every iteration, increase the layer count with each layer having nodes found above.
RMSE_arr=[]
iter_arr=[]
iter_arr=np.arange(1, 100, 10)
for i in np.arange(1, 100, 10):
	layer.append(iter_min)
	print(layer)
	nnreg = neural_network.MLPRegressor(hidden_layer_sizes=layer)
	nnreg.fit(normalized_X, Y_train.values)
	predicted_test = nnreg.predict(normalized_X_test)
	RMSE_new = np.sqrt(metrics.mean_squared_error(Y_test, predicted_test))
	print(("Root Mean Squared Error: ({}), iter: ({})\n").format(RMSE_new, i))
	RMSE_arr.append(RMSE_new)
	#print(nnreg.coefs_)
	#print("\nFeature vs coefficient strength: \n")
	coef_strength = pd.DataFrame(list(zip(X_train.columns, nnreg.coefs_)), columns=["features", "estimatedCoefficients"])
	print(coef_strength)
	i=i+1
	
nodes_min = iter_min
RMSE_min = np.min(RMSE_arr)
iter_min = np.argmin(RMSE_arr)
#Plot RMSE vs #Nodes for a single layer
plt.scatter(iter_arr, RMSE_arr)
plt.scatter(iter_arr[iter_min], RMSE_min, color = 'g')
plt.xlabel(("No. of Layers with each having {} nodes").format(nodes_min), fontsize=16)
plt.ylabel("RMSE", fontsize=16)
plt.title(("RMSE vs Layers # -- min RMSE = {} at {} layers").format(RMSE_min, iter_arr[iter_min]), fontsize=20)
plt.grid()
plt.show()

print("Root Mean Squared Error: %.5f\n" % RMSE)		
#Scatter plot -- predicted vs observed
fig1 = plt.figure(1)
frame1 = fig1.add_axes((.1,.3,.8,.6))
#plt.scatter(X_test, Y_test)
plt.scatter(predicted_test, Y_test)
plt.plot(predicted_test, predicted_test, '-r')
plt.ylabel("Observed Size of Backup (GB)")
frame1.set_xticklabels([])
plt.grid()
plt.title(("Results after fitting a Neural Network Regressor, RMSE = {}").format(RMSE_min), fontsize=20)
#plt.show()

#Residual Plots
frame2 = fig1.add_axes((.1,.1,.8,.2))
#plt.scatter(predicted_train, Y_train-predicted_train, c='b', s=40, alpha=0.3)
plt.scatter(predicted_test, Y_test-predicted_test)
plt.hlines(y=0, xmin=0, xmax=1)
plt.ylabel("Residual", fontsize=16)
plt.xlabel("Fitted Value - Size of Backup (GB)", fontsize=16)
plt.grid()
plt.show()