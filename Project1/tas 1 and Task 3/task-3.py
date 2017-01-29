import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation  

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

    lm = linear_model.LinearRegression()
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.10, random_state = 42)
    #note: put random_State as 42  no clue what that shit means
    #please help 
    lm.fit(X_train,Y_train)
    Y_predicted = lm.predict(X_test)
    rmse = np.sqrt(np.mean((Y_predicted - Y_test) ** 2))
    print str(rmse)  + " : workflow" + str(wid)
    #RSME data 
    #0.0282738461857 : workflow0
    #0.0270275556693 : workflow2
    #0.00605394655862 : workflow3
    #0.0940149621127 : workflow1
    #0.0836856164935 : workflow4

    #plt.scatter(Y_test, Y_predicted)
    #plt.figure(1)
    #plt.xlabel("Actual Median Value")
    #plt.ylabel("Predicted Median Value")
    #plt.title('Fitted values vs Actual Values Workflow :'  + str(wid))
    #plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], lw=5)
    #plt.show()

    plt.figure(1)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values plot Workflow :'  + str(wid))
    plt.scatter(Y_predicted, Y_predicted - Y_test)
    plt.hlines(y=0, xmin=-20, xmax=100)
    plt.show()

    