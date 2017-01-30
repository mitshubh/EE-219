import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model
from sklearn.model_selection import KFold, cross_val_score, train_test_split
############################################################################################
#Task 3
daysOfTheWeek = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
data = pd.read_csv("network_backup_dataset.csv")
data['Day of Week'] = data['Day of Week'].apply(lambda x:daysOfTheWeek[x])
data['Work-Flow-ID'] = data['Work-Flow-ID'].apply(lambda x: x[-1])
data['File Name'] = data['File Name'].apply(lambda x: x.rsplit('_', 1)[1])
for wid in pd.unique(data['Work-Flow-ID']):
    new_data = data[data['Work-Flow-ID'] == wid]
    Y = new_data['Size of Backup (GB)']
    X = new_data.drop('Size of Backup (GB)',axis = 1)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=50)
	
	#normalize
	normalized_X = preprocessing.normalize(X_train)
	normalized_X_test = preprocessing.normalize(X_test)
	
	model.fit(normalized_X, Y_train.values)
	
    lm = linear_model.LinearRegression()
    lm.fit(X_train,Y_train)
    Y_predicted = lm.predict(X_test)
    rmse = np.sqrt(np.mean((Y_predicted - Y_test) ** 2))
    print str(rmse)  + " : workflow" + str(wid)
   
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


    