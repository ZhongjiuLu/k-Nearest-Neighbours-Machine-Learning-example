#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:31:50 2017

@author: Zhongjiulu
"""

# import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.cross_validation import cross_val_score
import warnings; warnings.simplefilter('ignore')
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# loading the data set
wine = pd.read_csv('wine-quality-red.csv',delimiter=";")

# Create a binary variable "good wine", I define "good wine" if the quality is 6 or above, otherwise not.
wine["good wine"] = np.where(data["quality"]>=6, 1, 0)

# Convert data type of int64 into category data
wine["good wine"] = wine["good wine"].astype("category")



# split the data set into a training and test set (50/50)

# select the predictor variables except "quality" and "good wine"
x = data.iloc[:,:-2]

# response variable - "good wine"
y = data.iloc[:,-1:]

# Firstly shuffle the data,then split it (50/50)
train_x, test_x, train_y, test_y = train_test_split(x, y, 
                                                    test_size=0.5,random_state=42,
                                                    shuffle=True)

# normalises the data using Z-score transform

# create empty dataframe to store data
train_x_normalised = pd.DataFrame()
test_x_normalised = pd.DataFrame()

# looping through each observation to complete z-score transformation
for col in train_x:
    train_x_normalised[col] = (train_x[col]-train_x[col].mean())/train_x[col].std()

for col in test_x:
    test_x_normalised[col] = (test_x[col]-train_x[col].mean())/train_x[col].std()


#k-Nearest Neighbours classifiers for k = 1, 6, 11, ... 500
k_values = np.arange(1, 500, 5)

#appends 500 that would be otherwise missing
np.append(k_values,500)

# Empty dictionary for accuracy scores to be added
accuracy_scores = {}
    
for k in k_values:
    # instantiate learning model for every k value
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # fitting the model (the real learning part)
    knn.fit(train_x_normalised, train_y)
    
    # predicts the response
    pred = knn.predict(test_x_normalised)
    
    #adds accuracy score 
    accuracy_scores[k] = accuracy_score(test_y, pred)
    
# evaluates each classifier using 5-fold cross validation and select the best one
# create a empty list for storing the cross validation values
cv_scores = []

# perform 5-fold cross validation
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, train_x_normalised.as_matrix(), 
                             train_y.as_matrix().ravel(), cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    
# misclassification rate in each model    
MSE = [1 - x for x in cv_scores]

# find the best k
optimal_k = k_values[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# graph of misclassification error vs k value
plt.plot(k_values, MSE)
plt.xlabel('Number of K')
plt.ylabel('Misclassification Rate')
plt.show()

# Predicts the generalisation error using the test data set, as well as confusion matrix

# instantiate learning model
knn = KNeighborsClassifier(n_neighbors = optimal_k)

# fits the model
knn.fit(train_x_normalised, train_y)

# predicts the response
pred = knn.predict(test_x_normalised)

#creates confusion matrix
conf_matrix = confusion_matrix(test_y, pred)

#prints confusion matrix as a dataframe
print(pd.DataFrame(conf_matrix))

# Better visualization of confusion matrix
def plot_confusion_matrix(cm, target_names, 
                          title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(conf_matrix, ['0', '1'])
plt.show()






