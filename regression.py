import csv
import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dataset = sys.argv[1] if len(sys.argv) > 1 else "Bike-Sharing-Dataset/day.csv"

X = []
Y = []


columns = []

X_columns = ["workingday", "atemp","hum","windspeed"] # TODO(Bieber): Turn weekday into multiple indicator variables
Y_columns = ["cnt"]

X_indices = []
Y_indices = []

with open(dataset, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for i,row in enumerate(reader):
        if i == 0: # header
            columns = row
            for j, col in enumerate(row):
                if col in X_columns:
                    X_indices.append(j)
                if col in Y_columns:
                    Y_indices.append(j)
        else:
            x = []
            y = []

            for j, data in enumerate(row):
                if columns[j] == "weekday":
                    day = int(data)
                    data = [0] * 6
                    if day != 6:
                        data[day] = 1
                elif j in X_indices or j in Y_indices:
                    data = [float(data)]

                if j in X_indices:
                    x.extend(data)
                if j in Y_indices:
                    y.extend(data)

            X.append(x)
            Y.append(y)

clf = linear_model.LinearRegression()
# clf = linear_model.Ridge(alpha = .5)
clfLasso = linear_model.Lasso(alpha = .2)

print clf.fit(X, Y)
print clf.coef_
print clf.intercept_

print clfLasso.fit(X, Y)
print clfLasso.coef_

colors = ['r', 'g', 'b']

for i in range(len(X_columns)):
    plt.figure(i)
    plt.scatter([x[i] for x in X], [y[0] for y in Y], c=colors[i])
    plt.plot([x[i] for x in X], [clf.coef_[0][i]*x[i] for x in X], color='b')

plt.show()
