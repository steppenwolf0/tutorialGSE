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
	dfData = read_csv("./data/data.csv", header=None, sep=',', dtype=float).values
	dfLabels = read_csv("./data/labels.csv", header=None)
	biomarkers = read_csv("./data/features_0.csv", header=None)
	dfData=np.transpose(dfData)
	return dfData, dfLabels.values.ravel(), biomarkers.values.ravel() # to have it in the format that the classifiers like

def featureSelection() :
	
	#load Dataset
	X_0, y, biomarkerNames = loadDataset()
	
	for i in (2,4,8,16):
		#declare selector with 4 features using F-score
		selector=SelectKBest(f_classif, k=i)
		#Normalize Data
		scaler = StandardScaler()
		X = scaler.fit_transform(X_0)
		#Calculate Scores
		X_new = selector.fit_transform(X, y)
		#Get positions of Best Scores
		selected=selector.get_support(indices=True)
		##Print ANOVA F-Values
		#print("ANOVA F-value")
		#print(selector.scores_[selected])
		##Print P-values
		#print("p values")
		#print(selector.pvalues_[selected])
		##Print Resulting Features
		#print("features names")
		#print(biomarkerNames[selected])
		#print("features index")
		##Print Features Index
		#print(selected)
		print(i)
		#Declare Classifier
		clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)
		#Train Classifier
		clf.fit(X_new, y)
		#Print Accuracy
		print(clf.score(X_new,y))
	
		## create folder
		#folderName ="./results/"
		#if not os.path.exists(folderName) : os.makedirs(folderName)
		##Print reduce Dataset
		#pd.DataFrame(X_new).to_csv(folderName+"data_"+str(0)+".csv", header=None, index =None)
		#pd.DataFrame(biomarkerNames[selected]).to_csv(folderName+"features_"+str(0)+".csv", header=None, index =None)
		#pd.DataFrame(y).to_csv(folderName+"labels.csv", header=None, index =None)
	return 

if __name__ == "__main__" :
	sys.exit( featureSelection() )