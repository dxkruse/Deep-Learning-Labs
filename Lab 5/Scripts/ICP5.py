# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 21:32:25 2021

@author: Dietrich
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#%% Problem 1

# Read in data
train = pd.read_csv('C:/Users/Dietrich/Documents/GitHub/Deep-Learning-Labs/Lab 5/Scripts/data.csv')

# Handle missing values
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

# Delete rows where GarageArea == 0, meaning there was either no garage
# or GarageArea was not properly recorded.
data = data[data.GarageArea != 0]

##Build a linear model
y = np.log(data.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)


# Plot Sale Price vs Garage Area to check for outliers
plt.scatter(data.GarageArea,y)
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Sale Price vs Garage Area')

#%% Problem 2 and 3

# Read in data
train = pd.read_csv('C:/Users/Dietrich/Documents/GitHub/Deep-Learning-Labs/Lab 5/Scripts/data2.csv')
# Initialize Plot
plt.style.use(style = 'ggplot')
plt.rcParams['figure.figsize'] = (10,6)

#Convert Categorical data to numerical
train['City Group'] = train['City Group'].map( {'Big Cities': 1, 'Other': 0} ).astype(int)
train['Type'] = train['Type'].map( {'FC': 0, 'IL': 1, 'DT': 2, 'MB':3} ).astype(int)

# Get stats for revenue column and check skew
print(train.revenue.describe())
print(train.revenue.skew())
#plt.hist(train.revenue)
plt.show()
y = np.log(train.revenue)
print('skew is', y.skew())
plt.hist(y)


#%%
# Check correlation
numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
top_corr = corr['revenue'].sort_values(ascending=False)[:6]
print(corr['revenue'].sort_values(ascending=False)[:6],'\n')
print(corr['revenue'].sort_values(ascending=False)[-5:])




#%% Model Creation

# Initialize X and Y data
y = np.log(train.revenue)
#X = train.drop(['revenue', 'Id'], axis=1)
X = train[top_corr.index.drop('revenue')]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
# Perform Linear Regression
lr = linear_model.LinearRegression()
# Fit model to training data
model = lr.fit(X_train, y_train)
# Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

#%%
##visualize

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75, color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
