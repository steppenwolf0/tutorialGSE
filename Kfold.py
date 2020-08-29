import copy
import datetime
import logging
import numpy as np
import os
import sys

# used for normalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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
	X_0, y, biomarkerNames = loadDataset()\
	#use K-Fold
	kf = KFold(n_splits=10)
	kf.get_n_splits(X_0)
	
	for i in (250,500,1000):
		print("Number of Features "+str(i))
		fold=0
		for train_index, test_index in kf.split(X_0):
			print("Fold "+str(fold))
			fold=fold+1
			#declare selector with 4 features using F-score
			selector=SelectKBest(f_classif, k=i)
			#Normalize Data
			scaler = StandardScaler()
			X_train, X_test = X_0[train_index], X_0[test_index]
			y_train, y_test = y[train_index], y[test_index]
			X_train = scaler.fit_transform(X_train)
			X_test=scaler.transform(X_test)
			#Calculate Scores
			X_train = selector.fit_transform(X_train, y_train)
			#Get positions of Best Scores
			selected=selector.get_support(indices=True)
			X_test=selector.transform(X_test)
			##Print ANOVA F-Values
			#print("ANOVA F-value")
			#print(selector.scores_[selected])
			##Print P-values
			#print("p values")
			#print(selector.pvalues_[selected])
			#Print Resulting FeaturesS
			#print("features names")
			#print(biomarkerNames[selected])
			#print("features index")
			##Print Features Index
			#print(selected)
			#Declare Classifier
			clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)
			#Train Classifier
			clf.fit(X_train, y_train)
			#Print Accuracy
			accuracy_train=clf.score(X_train,y_train)
			accuracy_test=clf.score(X_test,y_test)
			print("Accuracy Train " + str(accuracy_train))
			print("Accuracy Test " + str(accuracy_test))
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