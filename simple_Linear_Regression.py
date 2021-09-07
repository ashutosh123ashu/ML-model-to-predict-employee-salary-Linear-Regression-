# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 23:38:42 2021

@author: hp
""" 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the dataset into the Training set andn Train set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state =0)

# Fitting Simple Linear Regression to the training class
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test results
Y_predicted = regressor.predict(X_test)
# Note: Y_predicted is the predicted salary and Y_test is the actual salary.

#Visualizing the training set results
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
#plt.show()

#Visualizing the Test set results
plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
#plt.show()