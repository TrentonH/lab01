__author__ = 'Trenton'

#part 1
#bro. Burton code
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

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

    def predictCar(self, X_test):
        y_test = []
        for i in X_test:
            dist = []
            for j in self.X_train:
                dist.append(sum([(k - l)**2 for k,l in zip(i, j)]))#collect the distances put in array
            small = list(zip(dist, y_train))#fined the k smallest distances
            small.sort(key= getKey)
            A = small[:self.n_neighbors]
            targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for n in A:
                index = int(n[1])
                targets[index] += 1
            maxamum = max(targets)
            maxIndex = [i for i, j in enumerate(targets) if j == maxamum]
            if(len(maxIndex) == 1):
                y_test.append(maxIndex[0])
            else:
                y_test.append(random.choice(maxIndex))
        return np.array(y_test)


class KNNClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    def fit(self, X_train, y_train):
        return KNNModle(X_train, y_train, self.n_neighbors)

#classifier = KNNClassifier(n_neighbors=1)
#model = classifier.fit(X_train, y_train)
#targets_predicted = model.predict(X_test)


#acc = accuracy_score(y_test,targets_predicted)
#print ("The prcent correct is {0:.2%}".format(acc))
#print (targets_predicted)


############part3##########

print("part 3 starts here")

# Use the k-Nearest-Neighbors algorithm to
# accurately train a model based on the data
# and targets
def createModel(data_train, targets_train):
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(data_train, targets_train)
    return model

# Use the model to predict the targets
def modelPredict(model, data_test):
    targets_predicted = model.predict(data_test)
    return targets_predicted

# Display the accuracy of the real KNN model
def displayAccuracy(targets_predicted, targets_test):
    accuracy = accuracy_score(targets_test, targets_predicted)
    size = len(targets_test)
    matches = int(accuracy * size)
    print("k-Nearest-Neighbors Accuracy: {0:.2%}".format(accuracy))
    print("k-Nearest-Neighbors Matches: {} out of {}".format(matches, size))
    return matches
def getAccuracy(targets_predicted, targets_test):
    return accuracy_score(targets_test, targets_predicted)



#pulls in the data and gives it columns
DA = pd.io.parsers.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",header=None)
DA.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "acc"]
obj_def = DA.select_dtypes(include=['object']).copy()

#changes the data to numbers insted of strings
DAchange = {"buying": {"vhigh": 10, "high": 7, "med": 4, "low": 0},
            "maint": {"vhigh": 10, "high": 7, "med": 4, "low": 0},
            "doors": {"5more": 5, "2": 2, "3": 3, "4": 4},
            "persons": {"more": 5, "2": 2, "4": 4},
            "lug_boot": {"small": 0, "med": 5, "big": 10},
            "safety": {"low": 0, "med": 5, "high": 10},
            "acc": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}}

#replae the data to what i said
obj_def.replace(DAchange, inplace=True)
CarNP = obj_def.as_matrix()

carTargets =[]
for x in CarNP:
    carTargets.append(x[6])
CarNP = np.delete(CarNP, 6, 1)

dataNP = CarNP
targetNP = np.array(carTargets)

#X_train, X_test, y_train, y_test = train_test_split(CarNP, carTargets, test_size=0.3)

totalAccuracy = 0
myAccuracy = 0

kf = KFold(n_splits=10)
for train, test in kf.split(CarNP):
    X_train, X_test, y_train, y_test = dataNP[train], dataNP[test], targetNP[train], targetNP[test]
    classifier = KNNClassifier(n_neighbors=1)
    model = classifier.fit(X_train, y_train)
    targets_predicted = model.predictCar(X_test)
    myAccuracy += accuracy_score(y_test,targets_predicted)
    m1 = createModel(X_train, y_train)
    tp = modelPredict(m1, X_test)
    totalAccuracy += getAccuracy(tp, y_test)
print ("kfold off the shelf prcent correct is ")
print((totalAccuracy/kf.get_n_splits()*100))
print ("my kfold prcent correct is ")
print((myAccuracy/kf.get_n_splits()*100))



###############################################
#autoHeaders =
#DC = pd.read_csv("auto-mpg.data",header=None, names=autoHeaders )

#DC["num_cylinders"].value_counts()
#print(DC.head(10))
###############################################
#dataPima = pd.read_csv("pima-indians-diabetes.data",header=None, names=pimaHeaders,  na_values=[" ?"] )
