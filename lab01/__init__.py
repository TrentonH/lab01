__author__ = 'Trenton'

#part 1
#bro. Burton code
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import csv
from sklearn.neighbors import KNeighborsClassifier

#extra credit
#spamReader = csv.reader(open('eggs.csv', newline=''), delimiter=' ', quotechar='|')


iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

#################part2#########################
print("part 2!!!!!!!!!!!!!!!!")

import numpy as np


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)


#part3

classifier = GaussianNB()
model = classifier.fit(X_train, y_train)

#part4
targets_predicted = model.predict(X_test)

#part5

class HardCodedClassifier:
    def fit(self, X_train, y_train):
        return HardModle()
class HardModle:
    def predict(self, X_test):
        return np.array([0 for n in X_test])





classifier = HardCodedClassifier()
model = classifier.fit(X_train, y_train)
targets_predicted = model.predict(X_test)

acc = accuracy_score(y_test,targets_predicted)
print ("The prcent correct is {0:.2%}".format(acc))


print (targets_predicted)


################################################


print("lab2 starts here ")

def getKey(item):
    return item[0]

class KNNModle:
    def __init__(self,X_train, y_train, n_neighbors):
        self.X_train = X_train
        self.y_train = y_train
        self.n_neighbors = n_neighbors
    def predict(self, X_test):
        y_test = []
        for i in X_test:
            dist = []
            g1 = 0
            g2 = 0
            g3 = 0
            for j in self.X_train:
                dist.append(sum([(k - l)**2 for k,l in zip(i, j)]))#collect the distances put in array
            small = list(zip(dist, y_train))#fined the k smallest distances
            small.sort(key= getKey)
            A = small[:self.n_neighbors]
            for x in A:#i want the second element not the first
                if x[1] == 0:
                    g1 = g1 + 1
                elif x[1] == 1:
                    g2 = g2 + 1
                else:
                    g3 = g3 + 1
            if g1 > g2 and g1 > g3:
                y_test.append(0)
            elif g2 > g1 and g2 > g3:
                y_test.append(1)
            elif g3 > g1 and g3 > g2:
                y_test.append(2)
            else:
                y_test.append(0)
        return np.array(y_test)


class KNNClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    def fit(self, X_train, y_train):
        return KNNModle(X_train, y_train, self.n_neighbors)

classifier = KNNClassifier(n_neighbors=3)
model = classifier.fit(X_train, y_train)
targets_predicted = model.predict(X_test)


acc = accuracy_score(y_test,targets_predicted)
print ("The prcent correct is {0:.2%}".format(acc))
print (targets_predicted)