__author__ = 'Trenton'

#part 1
#bro. Burton code
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import csv

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
        return self
    def predict(self, X_test):
        return np.array([0 for n in X_test])


classifier = HardCodedClassifier()
model = classifier.fit(X_train, y_train)
targets_predicted = model.predict(X_test)

acc = accuracy_score(y_test,targets_predicted)
print ("The prcent correct is {0:.2%}".format(acc))


print (targets_predicted)