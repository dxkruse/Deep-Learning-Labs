# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#%% Problem 1

# Read CSV
df = pd.read_csv('train_preprocessed.csv')

# Analysis using group by mean method
print(df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False), '\n')

# My own analysis with using a correlation coefficient
# Calculate Correlation Coefficient between Survived and Sex columns
correlation_matrix = np.corrcoef(df['Survived'], df['Sex'])
r_squared = correlation_matrix[0,1]**2
print("Correlation Coefficient: ", r_squared)


#%% Problem 2

# Using Naive Bayes to create a model and predict glass types

df = pd.read_csv('glass.csv')

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(df[['RI', 'Na', 'Mg', 'Al', 'Si','K','Ca','Ba','Fe']], df['Type'], test_size=0.4, random_state=0)

# Initialize and fit model to training data
model = GaussianNB()
model.fit(X_train, Y_train)

# Use model to predict output based on test data
Y_pred = model.predict(X_test)

#Calculate score and classification report
nb_score1 = sklearn.metrics.accuracy_score(Y_test, Y_pred)
nb_score2 = round(model.score(X_train, Y_train) * 100, 2)
nb_report = sklearn.metrics.classification_report(Y_test, Y_pred)
print("Naive Bayes Accuracy: ", nb_score1)
print("Classification Report: \n", nb_report)

#%% Problem 3

# Initialize and fit SVM model to training data
svc = SVC()
svc.fit(X_train, Y_train)

# Use model to predict output based on test data
Y_pred = svc.predict(X_test)

# Calculate score and classification report
svm_score1 = sklearn.metrics.accuracy_score(Y_test, Y_pred)
svm_score2 = round(svc.score(X_train, Y_train) * 100, 2)
svm_report = sklearn.metrics.classification_report(Y_test, Y_pred)
print("SVM Accuracy:", svm_score1)
print("Classification Report: \n", svm_report)

