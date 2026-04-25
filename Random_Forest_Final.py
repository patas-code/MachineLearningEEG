# importing dataset
path = 'AD_all_patients.csv'


#import all libraries necessary for random forest regression
import os
import random
import numpy as np
import matplotlib

import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from IPython.display import Image


#dataset parsing
main=pd.read_csv(path)
x=main
y=main.status

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#feature engineering -> look at importance on WIP file, normalize here and check balance after



#random forest & testing
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#confusion matrix

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
#exit figure yay

# feature importance graphic
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).plot.bar()
