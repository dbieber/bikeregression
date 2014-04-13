import csv
import sys
import numpy as np
from sklearn import linear_model
import pylab as pl

dataset = sys.argv[1] if len(sys.argv) > 1 else "Bike-Sharing-Dataset/day.csv"

X = []
Y = []

X_columns = ["atemp","hum","windspeed"] # TODO(Bieber): Turn weekday into multiple indicator variables
Y_columns = ["cnt"]

X_indices = []
Y_indices = []

with open(dataset, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for i,row in enumerate(reader):
        if i == 0: # header
            for j, col in enumerate(row):
                if col in X_columns:
                    X_indices.append(j)
                if col in Y_columns:
                    Y_indices.append(j)
        else:
            X.append([float(row[j]) for j in X_indices])
            Y.append([float(row[j]) for j in Y_indices])

clf = linear_model.LinearRegression()
# clf = linear_model.Ridge(alpha = .5)
# clf = linear_model.Lasso(alpha = .2)

print clf.fit(X, Y)
print clf.coef_


