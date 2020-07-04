import copy
import datetime
import logging
import numpy as np
import os
import sys

# used for normalization
from sklearn.preprocessing import StandardScaler

# this is an incredibly useful function
from pandas import read_csv
import pandas as pd
#import feature selection with SelectKBest
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, mutual_info_classif, mutual_info_regression
#import a Classifier
from sklearn.linear_model import PassiveAggressiveClassifier

#load data
def loadDataset() :
	dfData = read_csv("./results/data_0.csv", header=None, sep=',', dtype=float).values
	dfLabels = read_csv("./results/labels.csv", header=None)
	biomarkers = read_csv("./results/features_0.csv", header=None)
	return dfData, dfLabels.values.ravel(), biomarkers.values.ravel() # to have it in the format that the classifiers like

def featureSelection() :
	#load Dataset
	X, y, biomarkerNames = loadDataset()
	#Normalize Data
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	#Declare Classifier
	clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)
	#Train Classifier
	clf.fit(X, y)
	#Print Accuracy
	print(clf.score(X,y))
	
	return 

if __name__ == "__main__" :
	sys.exit( featureSelection() )