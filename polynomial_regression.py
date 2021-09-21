# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 07:02:21 2021

@author: hp
"""
#Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the data set

dataset = pd.read_csv('Position_Salaries.csv')

# Creating the matrix of the dataset
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values


#Splitting the dataset into Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, 
                                                    train_size = 0.8,random_state=0)"""

#Fitting Linear Regression model to our data set
from sklearn.linear_model import LinearRegression
linearregressor = LinearRegression()
linearregressor.fit(X, Y)

#Fitting Polynomial Regression to our data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X,Y)
poly_reg.fit(X_poly, Y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y)

#Visualizing the result of linear regression result
plt.scatter(X, Y, color = 'red')
plt.plot(X, linearregressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

#Visualizing the result of polynomial regression result
plt.scatter(X, Y, color = 'red')
plt.plot(X,lin_reg.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

#Predicting a new result with linear regression
linearregressor.predict([6.5])

#Predicting a new result with polynomial regression model
lin_reg.predict(6.5)











