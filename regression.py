import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, preprocessing, cross_validation

dataset = sys.argv[1] if len(sys.argv) > 1 else "Bike-Sharing-Dataset/day.csv"

X = []
Y = []


columns = []

X_columns = ["yr", "workingday", "hum","windspeed", "atemp", "season"]
prettyNames = ["year", "workingday", "humidity", "windspeed", "temperature", "temperature^2", "temperature^3", "winter", "spring", "summer"]
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
                elif columns[j] == "atemp":
                    data = [float(data), float(data)**2, float(data)**3]
                elif columns[j] == "season":
                    season = int(data)
                    data = [0] * 4
                    if season != 4:
                        data[season - 1] = 1
                elif j in X_indices or j in Y_indices:
                    data = [float(data)]

                if j in X_indices:
                    x.extend(data)
                if j in Y_indices:
                    y.extend(data)

            X.append(x)
            Y.append(y)

# clf = linear_model.LinearRegression()
# clf = linear_model.Ridge(alpha = .5)
# clfLasso = linear_model.Lasso(alpha = .2)

X = np.array(X)
Y = np.array(Y)[:,0]

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, random_state=43)

clf = linear_model.LinearRegression()
fit = clf.fit(X_train, Y_train)
print "unconstrained:"
print fit.coef_
print ""

kf = cross_validation.KFold(len(Y_train), n_folds=10, shuffle=True)
clfLassoCV = linear_model.LassoLarsCV(cv=kf)
lassoCV = clfLassoCV.fit(X_train, Y_train)
print "lasso:"
print lassoCV.alpha_
print lassoCV.coef_
print ""

clfRidgeCV = linear_model.RidgeCV(cv=kf)
ridgeCV = clfRidgeCV.fit(X_train, Y_train)
print "ridge:"
print ridgeCV.alpha_
print ridgeCV.coef_
print ""

print "Scores (train):"
print "unconstrained: ", fit.score(X_train, Y_train)
print "lasso: ", lassoCV.score(X_train, Y_train)
print "ridge: ", ridgeCV.score(X_train, Y_train)
print ""

print "Scores (test):"
print "unconstrained: ", fit.score(X_test, Y_test)
print "lasso: ", lassoCV.score(X_test, Y_test)
print "ridge: ", ridgeCV.score(X_test, Y_test)

def plotLassoCoeffs():
    numVars = len(X_train[0])
    maxLen = 400
    coeffs = np.array([linear_model.Lasso(alpha=a).fit(X_train,Y_train).coef_ for a in range(1,maxLen + 1)])
    filteredCoeffs = [coeffs[abs(coeffs[:,i]) > 0,i] for i in range(numVars)]

    plots = [None] * numVars

    for i in range(numVars):
        if filteredCoeffs[i].shape[0] < maxLen:
            plots[i], = plt.semilogx(np.append(filteredCoeffs[i], 0))
        else:
            plots[i], = plt.semilogx(filteredCoeffs[i])

    plt.axvline(x=lassoCV.alpha_, color='k')
    #plt.axis([1,1000,-4000,8000])
    plt.gca().set_xlim([1,1000])

    plt.legend(plots, prettyNames, loc='upper right', ncol=3, bbox_to_anchor=(1.5, 1.05))
    plt.xlabel("alpha (l1 norm parameter)")
    plt.ylabel("weight")

def plotRidgeCoeffs():
    numVars = len(X_train[0])
    maxLen = 1000
    coeffs = np.array([linear_model.Ridge(alpha=a).fit(X_train,Y_train).coef_ for a in range(1,maxLen + 1)])
    filteredCoeffs = [coeffs[abs(coeffs[:,i]) > 0,i] for i in range(numVars)]

    plots = [None] * numVars

    for i in range(numVars):
        if filteredCoeffs[i].shape[0] < maxLen:
            plots[i], = plt.semilogx(np.append(filteredCoeffs[i], 0))
        else:
            plots[i], = plt.semilogx(filteredCoeffs[i])

    plt.axvline(x=ridgeCV.alpha_, color='k')
    #plt.axis([1,1000,-4000,8000])
    plt.gca().set_xlim([1,1000])

    plt.legend(plots, prettyNames, loc='upper right', ncol=3, bbox_to_anchor=(1.5, 1.05))
    plt.xlabel("alpha (l2 norm parameter)")
    plt.ylabel("weight")

plt.figure(1)
plotLassoCoeffs()
plt.figure(2)
plotRidgeCoeffs()

"""
normX = preprocessing.normalize(X, norm='l1', axis=0)

print clf.fit(X, Y)
print clf.coef_
print clf.intercept_

print clfLasso.fit(X, Y)
print clfLasso.coef_

plt.figure(1)
plt.scatter([x[3] for x in normX], [y[0] for y in Y], c=[x[0] for x in normX])
plt.figure(2)
plt.scatter([x[3] for x in X], [y[0] for y in Y], c=[x[0] for x in X])
plt.figure(3)
plt.scatter([x[3] for x in X], [x[-2]/float(x[-1]) for x in X], c=[x[0] for x in X])

colors = ['r', 'g', 'b', 'y']

for i in range(len(X_columns)):
    plt.figure(i)
    plt.scatter([x[i] for x in X], [y[0] for y in Y], c=colors[i])
    plt.plot([x[i] for x in X], clf.intercept_ + [clf.coef_[0][i]*x[i] for x in X], color='b')
"""
plt.show()

