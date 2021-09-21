# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:05:02 2021

@author: hp
"""
# SVR

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the data set

dataset = pd.read_csv('Position_Salaries.csv')

# Creating the matrix of the dataset
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values


#Splitting the dataset into Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, 
                                                    train_size = 0.8,random_state=0)"""

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)


#Fitting SVR to our data set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

#Predicting a new result with linear regression
y_pred = regressor.predict([[6.5]])
print(y_pred)

#Visualizing the SVR result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid =X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()














