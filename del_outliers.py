#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 12:40:35 2018

@author: Hana
"""

import pandas
from sklearn.ensemble import IsolationForest

# read X_train (after filling NAs and feature selection)
D1 = pandas.read_csv('data1.csv')
D1 = D1.iloc[:,1:] # delete ids
D1.drop(D1.columns[len(D1.columns)-1], axis=1, inplace=True) # delete last column y

D2 = pandas.read_csv('data2.csv')
D2 = D2.iloc[:,1:]
D2.drop(D2.columns[len(D2.columns)-1], axis=1, inplace=True)

# read y_train
Y = pandas.read_csv('y_train.csv')
Y = Y.iloc[:,1:] #delete ids

# a function to delete outliers using IsolationForest and write the filtered data
def del_outliers(D,contamination,filename):
    clf = IsolationForest(behaviour='new',contamination=contamination)
    clf.fit(D.iloc[:,1:])
    D_wo = clf.predict(D.iloc[:,1:])
    D_new = D[D_wo == 1]
    Y_new = Y[D_wo == 1]
    # add y to the last column of the dataframe
    D_new['y'] = Y_new
    D_new.to_csv(filename,index=False) 
    
del_outliers(D1,'auto',"wo-outliers_data1.csv")
del_outliers(D2, 'auto', "wo-outliers_data2.csv")

# Could set value to contamination to get certain percentage of remained samples
del_outliers(D1,0.1,"90data1.csv")