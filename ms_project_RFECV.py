import csv
import sys
import numpy as np

from sklearn.feature_selection import RFECV
from sklearn.svm import SVR


#read the csv file
reader = csv.reader(open('df1.csv', 'r'), delimiter = ",")
x = list(reader)
df = np.array(x)

#transpose the imported file
tdf = np.transpose(df)

#create labels
a = np.ones(25) ; b = np.zeros(71)
y = np.concatenate((a, b))

#features by removing patients' ID's, Gene ID's, Gene Ref, and column numbers 
X = tdf[3:, 1:]

#support vector machine with recursive feature elimination(SVM-RFE)
svm = SVR(kernel="linear")

rfe_cv = RFECV(estimator=svm, step = 500, cv = 50)

rfe_cv.fit(X, y)

np.savetxt("RFE_CV.csv", rfe_cv.ranking_, delimiter=",")
