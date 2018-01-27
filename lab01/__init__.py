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
            targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
print("cars starts here")

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



######################################################################################################################
######################################################################################################################
######################################################################################################################

print("pima starts here")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
dataset.columns = ["numPreg", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "age", "class"]

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

CarNP = dataset.as_matrix()


carTargets =[]
for x in CarNP:
    carTargets.append(x[8])
CarNP = np.delete(CarNP, 8, 1)

dataNP = CarNP
targetNP = np.array(carTargets)

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
######################################################################################################################
######################################################################################################################
######################################################################################################################
print("auto starts here")

dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", na_values="?", delim_whitespace=True,)
dataset.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "name"]

not_accepted = ['horsepower']

for column in not_accepted:
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

dataset["name"] = dataset["name"].replace (["ford pinto"], 1)
dataset["name"] = dataset["name"].replace (["ford maverick"], 2)
dataset["name"] = dataset["name"].replace (["amc matador"], 3)
dataset["name"] = dataset["name"].replace (["peugeot 504"], 4)
dataset["name"] = dataset["name"].replace (["amc hornet"], 5)
dataset["name"] = dataset["name"].replace (["chevrolet impala"], 6)
dataset["name"] = dataset["name"].replace (["amc gremlin"], 7)
dataset["name"] = dataset["name"].replace (["chevrolet chevette"], 8)
dataset["name"] = dataset["name"].replace (["toyota corona"], 9)
dataset["name"] = dataset["name"].replace (["ford galaxie 500"], 10)
dataset["name"] = dataset["name"].replace (["pontiac catalina"], 11)
dataset["name"] = dataset["name"].replace (["chevrolet citation"], 12)
dataset["name"] = dataset["name"].replace (["volkswagen dasher"], 13)
dataset["name"] = dataset["name"].replace (["ford gran torino"], 14)
dataset["name"] = dataset["name"].replace (["dodge colt"], 15)
dataset["name"] = dataset["name"].replace (["chevrolet vega"], 16)
dataset["name"] = dataset["name"].replace (["chevrolet nova"], 17)
dataset["name"] = dataset["name"].replace (["plymouth duster"], 18)
dataset["name"] = dataset["name"].replace (["plymouth fury iii"], 19)
dataset["name"] = dataset["name"].replace (["chevrolet caprice classic"], 20)
dataset["name"] = dataset["name"].replace (["honda civic"], 21)
dataset["name"] = dataset["name"].replace (["buick estate wagon (sw)"], 22)
dataset["name"] = dataset["name"].replace (["buick century"], 23)
dataset["name"] = dataset["name"].replace (["honda accord"], 24)
dataset["name"] = dataset["name"].replace (["toyota mark ii"], 25)
dataset["name"] = dataset["name"].replace (["ford ltd"], 26)
dataset["name"] = dataset["name"].replace (["mazda 626"], 27)
dataset["name"] = dataset["name"].replace (["chevrolet malibu"], 28)
dataset["name"] = dataset["name"].replace (["plymouth valiant"], 29)
dataset["name"] = dataset["name"].replace (["nissan stanza xe"], 30)
dataset["name"] = dataset["name"].replace (["plymouth horizon"], 31)
dataset["name"] = dataset["name"].replace (["amc concord d/l"], 32)
dataset["name"] = dataset["name"].replace (["datsun 1200"], 33)
dataset["name"] = dataset["name"].replace (["ford torino 500"], 34)
dataset["name"] = dataset["name"].replace (["ford granada ghia"], 35)
dataset["name"] = dataset["name"].replace (["mazda glc"], 36)
dataset["name"] = dataset["name"].replace (["pontiac j2000 se hatchback"], 37)
dataset["name"] = dataset["name"].replace (["oldsmobile cutlass ls"], 38)
dataset["name"] = dataset["name"].replace (["chevy c10"], 39)
dataset["name"] = dataset["name"].replace (["chrysler lebaron town @ country (sw)"], 40)
dataset["name"] = dataset["name"].replace (["toyota corolla 1600 (sw)"], 41)
dataset["name"] = dataset["name"].replace (["chrysler newport royal"], 42)
dataset["name"] = dataset["name"].replace (["maxda glc deluxe"], 43)
dataset["name"] = dataset["name"].replace (["chevrolet malibu classic (sw)"], 44)
dataset["name"] = dataset["name"].replace (["peugeot 505s turbo diesel"], 45)
dataset["name"] = dataset["name"].replace (["dodge aries se"], 46)
dataset["name"] = dataset["name"].replace (["renault 12 (sw)"], 47)
dataset["name"] = dataset["name"].replace (["bmw 2002"], 48)
dataset["name"] = dataset["name"].replace (["amc concord dl 6"], 49)
dataset["name"] = dataset["name"].replace (["buick electra 225 custom"], 50)
dataset["name"] = dataset["name"].replace (["plymouth volare custom"], 51)
dataset["name"] = dataset["name"].replace (["plymouth fury gran sedan"], 52)
dataset["name"] = dataset["name"].replace (["plymouth sapporo"], 53)
dataset["name"] = dataset["name"].replace (["oldsmobile starfire sx"], 54)
dataset["name"] = dataset["name"].replace (["ford fairmont"], 55)
dataset["name"] = dataset["name"].replace (["volkswagen rabbit l"], 56)
dataset["name"] = dataset["name"].replace (["chevrolet cavalier 2-door"], 57)
dataset["name"] = dataset["name"].replace (["chevy s-10"], 58)
dataset["name"] = dataset["name"].replace (["buick century 350"], 59)
dataset["name"] = dataset["name"].replace (["oldsmobile omega"], 60)
dataset["name"] = dataset["name"].replace (["chevrolet monte carlo landau"], 61)
dataset["name"] = dataset["name"].replace (["chrysler cordoba"], 62)
dataset["name"] = dataset["name"].replace (["amc ambassador brougham"], 63)
dataset["name"] = dataset["name"].replace (["mercury monarch ghia"], 64)
dataset["name"] = dataset["name"].replace (["ford mustang ii"], 65)
dataset["name"] = dataset["name"].replace (["datsun 610"], 66)
dataset["name"] = dataset["name"].replace (["ford granada l"], 67)
dataset["name"] = dataset["name"].replace (["toyota corolla"], 68)
dataset["name"] = dataset["name"].replace (["toyota corolla 1200"], 69)
dataset["name"] = dataset["name"].replace (["saab 99le"], 70)
dataset["name"] = dataset["name"].replace (["chevrolet chevelle malibu classic"], 71)
dataset["name"] = dataset["name"].replace (["dodge aspen"], 72)
dataset["name"] = dataset["name"].replace (["datsun pl510"], 73)
dataset["name"] = dataset["name"].replace (["oldsmobile cutlass salon brougham"], 74)
dataset["name"] = dataset["name"].replace (["subaru"], 75)
dataset["name"] = dataset["name"].replace (["honda civic cvcc"], 76)
dataset["name"] = dataset["name"].replace (["renault 18i"], 77)
dataset["name"] = dataset["name"].replace (["toyouta corona mark ii (sw)"], 78)
dataset["name"] = dataset["name"].replace (["toyota corona mark ii"], 79)
dataset["name"] = dataset["name"].replace (["datsun b210"], 80)
dataset["name"] = dataset["name"].replace (["oldsmobile cutlass supreme"], 81)
dataset["name"] = dataset["name"].replace (["ford mustang gl"], 82)
dataset["name"] = dataset["name"].replace (["fiat 124b"], 83)
dataset["name"] = dataset["name"].replace (["chrysler new yorker brougham"], 84)
dataset["name"] = dataset["name"].replace (["audi 4000"], 84)
dataset["name"] = dataset["name"].replace (["audi fox"], 86)
dataset["name"] = dataset["name"].replace (["volvo 144ea"], 87)
dataset["name"] = dataset["name"].replace (["ford futura"], 88)
dataset["name"] = dataset["name"].replace (["toyota corolla liftback"], 89)
dataset["name"] = dataset["name"].replace (["oldsmobile omega brougham"], 90)
dataset["name"] = dataset["name"].replace (["fiat 124 sport coupe"], 91)
dataset["name"] = dataset["name"].replace (["dodge aspen se"], 92)
dataset["name"] = dataset["name"].replace (["amc rebel sst"], 93)
dataset["name"] = dataset["name"].replace (["plymouth satellite sebring"], 94)
dataset["name"] = dataset["name"].replace (["dodge st. regis"], 95)
dataset["name"] = dataset["name"].replace (["dodge rampage"], 96)
dataset["name"] = dataset["name"].replace (["dodge colt hatchback custom"], 97)
dataset["name"] = dataset["name"].replace (["toyota corona liftback"], 98)
dataset["name"] = dataset["name"].replace (["plymouth satellite custom (sw)"], 99)
dataset["name"] = dataset["name"].replace (["toyota celica gt"], 100)
dataset["name"] = dataset["name"].replace (["dodge d200"], 101)
dataset["name"] = dataset["name"].replace (["toyota tercel"], 102)
dataset["name"] = dataset["name"].replace (["plymouth satellite custom"], 103)
dataset["name"] = dataset["name"].replace (["audi 100ls"], 104)
dataset["name"] = dataset["name"].replace (["amc concord"], 105)
dataset["name"] = dataset["name"].replace (["vw rabbit"], 106)
dataset["name"] = dataset["name"].replace (["ford gran torino (sw)"], 107)
dataset["name"] = dataset["name"].replace (["opel 1900"], 108)
dataset["name"] = dataset["name"].replace (["buick century special"], 109)
dataset["name"] = dataset["name"].replace (["oldsmobile cutlass ciera (diesel"], 110)
dataset["name"] = dataset["name"].replace (["cadillac eldorado"], 111)
dataset["name"] = dataset["name"].replace (["honda civic 1500 gl"], 112)
dataset["name"] = dataset["name"].replace (["hi 1200d"], 113)
dataset["name"] = dataset["name"].replace (["dodge colt m/m"], 114)
dataset["name"] = dataset["name"].replace (["datsun 810 maxima"], 115)
dataset["name"] = dataset["name"].replace (["dodge charger 2.2"], 116)
dataset["name"] = dataset["name"].replace (["chevrolet cavalier"], 117)
dataset["name"] = dataset["name"].replace (["mercury marquis"], 118)
dataset["name"] = dataset["name"].replace (["chevrolet monte carlo s"], 119)
dataset["name"] = dataset["name"].replace (["mercury zephyr"], 120)
dataset["name"] = dataset["name"].replace (["chevrolet chevelle malibu"], 121)
dataset["name"] = dataset["name"].replace (["datsun 310 gx"], 122)
dataset["name"] = dataset["name"].replace (["dodge dart custom"], 123)
dataset["name"] = dataset["name"].replace (["chevrolet concours"], 124)
dataset["name"] = dataset["name"].replace (["ford torino"], 125)
dataset["name"] = dataset["name"].replace (["mercury grand marquis"], 126)
dataset["name"] = dataset["name"].replace (["chevrolet vega 2300"], 127)
dataset["name"] = dataset["name"].replace (["mercury capri 2000"], 128)
dataset["name"] = dataset["name"].replace (["plymouth horizon 4"], 129)
dataset["name"] = dataset["name"].replace (["toyota carina"], 130)
dataset["name"] = dataset["name"].replace (["mercedes benz 300d"], 131)
dataset["name"] = dataset["name"].replace (["ford ltd landau"], 132)
dataset["name"] = dataset["name"].replace (["mazda glc custom l"], 133)
dataset["name"] = dataset["name"].replace (["cadillac seville"], 134)
dataset["name"] = dataset["name"].replace (["chevrolet woody"], 135)
dataset["name"] = dataset["name"].replace (["maxda rx3"], 136)
dataset["name"] = dataset["name"].replace (["dodge challenger se"], 137)
dataset["name"] = dataset["name"].replace (["pontiac phoenix"], 138)
dataset["name"] = dataset["name"].replace (["ford ranger"], 139)
dataset["name"] = dataset["name"].replace (["plymouth grand fury"], 140)
dataset["name"] = dataset["name"].replace (["amc pacer d/l"], 141)
dataset["name"] = dataset["name"].replace (["plymouth 'cuda 340"], 142)
dataset["name"] = dataset["name"].replace (["toyota cressida"], 143)
dataset["name"] = dataset["name"].replace (["mercedes-benz 240d"], 144)
dataset["name"] = dataset["name"].replace (["ford pinto (sw)"], 145)
dataset["name"] = dataset["name"].replace (["volkswagen jetta"], 146)
dataset["name"] = dataset["name"].replace (["pontiac lemans v6"], 147)
dataset["name"] = dataset["name"].replace (["dodge aries wagon (sw)"], 148)
dataset["name"] = dataset["name"].replace (["datsun 510 (sw)"], 149)
dataset["name"] = dataset["name"].replace (["mercury marquis brougham"], 150)
dataset["name"] = dataset["name"].replace (["peugeot 304"], 151)
dataset["name"] = dataset["name"].replace (["volkswagen type 3"], 152)
dataset["name"] = dataset["name"].replace (["ford mustang ii 2+2"], 153)
dataset["name"] = dataset["name"].replace (["mercury capri v6"], 154)
dataset["name"] = dataset["name"].replace (["datsun b-210"], 155)
dataset["name"] = dataset["name"].replace (["amc ambassador dpl"], 156)
dataset["name"] = dataset["name"].replace (["datsun f-10 hatchback"], 157)
dataset["name"] = dataset["name"].replace (["volkswagen rabbit custom"], 158)
dataset["name"] = dataset["name"].replace (["datsun 510"], 159)
dataset["name"] = dataset["name"].replace (["volkswagen super beetle"], 160)
dataset["name"] = dataset["name"].replace (["fiat 131"], 161)
dataset["name"] = dataset["name"].replace (["datsun 200sx"], 162)
dataset["name"] = dataset["name"].replace (["ford escort 2h"], 163)
dataset["name"] = dataset["name"].replace (["renault 12tl"], 164)
dataset["name"] = dataset["name"].replace (["mercury zephyr 6"], 165)
dataset["name"] = dataset["name"].replace (["datsun b210 gx"], 166)
dataset["name"] = dataset["name"].replace (["toyota starlet"], 167)
dataset["name"] = dataset["name"].replace (["datsun 510 hatchback"], 168)
dataset["name"] = dataset["name"].replace (["volkswagen rabbit"], 169)
dataset["name"] = dataset["name"].replace (["plymouth grand fury"], 170)
dataset["name"] = dataset["name"].replace (["volkswagen 411 (sw)"], 171)
dataset["name"] = dataset["name"].replace (["honda civic 1300"], 172)
dataset["name"] = dataset["name"].replace (["saab 99e"], 173)
dataset["name"] = dataset["name"].replace (["ford fairmont (man)"], 174)
dataset["name"] = dataset["name"].replace (["dodge d100"], 175)
dataset["name"] = dataset["name"].replace (["plymouth valiant custom"], 177)
dataset["name"] = dataset["name"].replace (["fiat 124 tc"], 178)
dataset["name"] = dataset["name"].replace (["renault 5 gtl"], 179)
dataset["name"] = dataset["name"].replace (["pontiac sunbird coupe"], 180)
dataset["name"] = dataset["name"].replace (["triumph tr7 coupe"], 181)
dataset["name"] = dataset["name"].replace (["amc matador (sw)"], 182)
dataset["name"] = dataset["name"].replace (["buick opel isuzu deluxe"], 183)
dataset["name"] = dataset["name"].replace (["ford granada gl"], 184)
dataset["name"] = dataset["name"].replace (["vokswagen rabbit"], 185)
dataset["name"] = dataset["name"].replace (["volvo 145e (sw)"], 186)
dataset["name"] = dataset["name"].replace (["amc pacer"], 187)
dataset["name"] = dataset["name"].replace (["volvo 245"], 188)
dataset["name"] = dataset["name"].replace (["plymouth volare premier v8"], 189)
dataset["name"] = dataset["name"].replace (["volkswagen 1131 deluxe sedan"], 190)
dataset["name"] = dataset["name"].replace (["mercedes-benz 280s"], 191)
dataset["name"] = dataset["name"].replace (["pontiac grand prix"], 192)
dataset["name"] = dataset["name"].replace (["volkswagen scirocco"], 193)
dataset["name"] = dataset["name"].replace (["volvo 264gl"], 194)
dataset["name"] = dataset["name"].replace (["chevrolet vega (sw)"], 195)
dataset["name"] = dataset["name"].replace (["datsun 210 mpg"], 196)
dataset["name"] = dataset["name"].replace (["dodge omni"], 197)
dataset["name"] = dataset["name"].replace (["mercury lynx l"], 198)
dataset["name"] = dataset["name"].replace (["amc hornet sportabout (sw)"], 199)
dataset["name"] = dataset["name"].replace (["plymouth fury"], 201)
dataset["name"] = dataset["name"].replace (["dodge magnum xe"], 202)
dataset["name"] = dataset["name"].replace (["amc ambassador sst"], 203)
dataset["name"] = dataset["name"].replace (["plymouth arrow gs"], 204)
dataset["name"] = dataset["name"].replace (["ford fairmont (auto)"], 205)
dataset["name"] = dataset["name"].replace (["volkswagen rabbit custom diesel"], 206)
dataset["name"] = dataset["name"].replace (["datsun 810"], 207)
dataset["name"] = dataset["name"].replace (["dodge aspen 6"], 208)
dataset["name"] = dataset["name"].replace (["chevrolet monte carlo"], 209)
dataset["name"] = dataset["name"].replace (["oldsmobile cutlass ciera (diesel)"], 210)
dataset["name"] = dataset["name"].replace (["vw rabbit custom"], 211)
dataset["name"] = dataset["name"].replace (["datsun 710"], 212)
dataset["name"] = dataset["name"].replace (["fiat 128"], 213)
dataset["name"] = dataset["name"].replace (["pontiac firebird"], 214)
dataset["name"] = dataset["name"].replace (["audi 100 ls"], 215)
dataset["name"] = dataset["name"].replace (["buick skylark limited"], 216)
dataset["name"] = dataset["name"].replace (["honda prelude"], 217)
dataset["name"] = dataset["name"].replace (["buick century limited"], 218)
dataset["name"] = dataset["name"].replace (["ford mustang"], 219)
dataset["name"] = dataset["name"].replace (["vw dasher (diesel)"], 220)
dataset["name"] = dataset["name"].replace (["ford pinto runabout"], 221)
dataset["name"] = dataset["name"].replace (["ford thunderbird"], 222)
dataset["name"] = dataset["name"].replace (["buick skylark"], 223)
dataset["name"] = dataset["name"].replace (["plymouth champ"], 224)
dataset["name"] = dataset["name"].replace (["pontiac astro"], 225)
dataset["name"] = dataset["name"].replace (["dodge colt hardtop"], 226)
dataset["name"] = dataset["name"].replace (["chevrolet chevelle concours (sw)"], 227)
dataset["name"] = dataset["name"].replace (["mazda rx-7 gs"], 228)
dataset["name"] = dataset["name"].replace (["mazda rx2 coupe"], 229)
dataset["name"] = dataset["name"].replace (["renault lecar deluxe"], 230)
dataset["name"] = dataset["name"].replace (["datsun 210"], 231)
dataset["name"] = dataset["name"].replace (["subaru dl"], 232)
dataset["name"] = dataset["name"].replace (["dodge diplomat"], 233)
dataset["name"] = dataset["name"].replace (["ford granada"], 234)
dataset["name"] = dataset["name"].replace (["mazda rx-4"], 235)
dataset["name"] = dataset["name"].replace (["volkswagen model 111"], 236)
dataset["name"] = dataset["name"].replace (["buick skylark 320"], 237)
dataset["name"] = dataset["name"].replace (["mercury cougar brougham"], 238)
dataset["name"] = dataset["name"].replace (["ford mustang cobra"], 239)
dataset["name"] = dataset["name"].replace (["ford fiesta"], 240)
dataset["name"] = dataset["name"].replace (["chevrolet bel air"], 241)
dataset["name"] = dataset["name"].replace (["toyota corona hardtop"], 242)
dataset["name"] = dataset["name"].replace (["oldsmobile vista cruiser"], 243)
dataset["name"] = dataset["name"].replace (["volvo 244dl"], 244)
dataset["name"] = dataset["name"].replace (["ford escort 4w"], 245)
dataset["name"] = dataset["name"].replace (["amc spirit dl"], 246)
dataset["name"] = dataset["name"].replace (["plymouth custom suburb"], 247)
dataset["name"] = dataset["name"].replace (["bmw 320i"], 248)
dataset["name"] = dataset["name"].replace (["buick century luxus (sw)"], 249)
dataset["name"] = dataset["name"].replace (["dodge coronet brougham"], 250)
dataset["name"] = dataset["name"].replace (["plymouth volare"], 251)
dataset["name"] = dataset["name"].replace (["pontiac ventura sj"], 252)
dataset["name"] = dataset["name"].replace (["vw rabbit c (diesel)"], 253)
dataset["name"] = dataset["name"].replace (["plymouth volare"], 254)
dataset["name"] = dataset["name"].replace (["dodge monaco brougham"], 255)
dataset["name"] = dataset["name"].replace (["opel manta"], 256)
dataset["name"] = dataset["name"].replace (["ford country squire (sw)"], 257)
dataset["name"] = dataset["name"].replace (["chevrolet nova custom"], 258)
dataset["name"] = dataset["name"].replace (["pontiac catalina brougham"], 259)
dataset["name"] = dataset["name"].replace (["pontiac phoenix lj"], 260)
dataset["name"] = dataset["name"].replace (["datsun 200-sx"], 261)
dataset["name"] = dataset["name"].replace (["dodge monaco (sw)"], 262)
dataset["name"] = dataset["name"].replace (["dodge coronet custom (sw)"], 263)
dataset["name"] = dataset["name"].replace (["ford fairmont futura"], 264)
dataset["name"] = dataset["name"].replace (["plymouth cricket"], 265)
dataset["name"] = dataset["name"].replace (["plymouth satellite"], 266)
dataset["name"] = dataset["name"].replace (["buick lesabre custom"], 267)
dataset["name"] = dataset["name"].replace (["buick skyhawk"], 268)
dataset["name"] = dataset["name"].replace (["plymouth reliant"], 269)
dataset["name"] = dataset["name"].replace (["oldsmobile delta 88 royale"], 270)
dataset["name"] = dataset["name"].replace (["mazda glc deluxe"], 271)
dataset["name"] = dataset["name"].replace (["honda civic (auto)"], 272)
dataset["name"] = dataset["name"].replace (["chevroelt chevelle malibu"], 273)
dataset["name"] = dataset["name"].replace (["mazda glc 4"], 274)
dataset["name"] = dataset["name"].replace (["dodge colt (sw)"], 275)
dataset["name"] = dataset["name"].replace (["ford country"], 276)
dataset["name"] = dataset["name"].replace (["vw pickup"], 277)
dataset["name"] = dataset["name"].replace (["chevrolet camaro"], 278)
dataset["name"] = dataset["name"].replace (["chrysler lebaron medallion"], 279)
dataset["name"] = dataset["name"].replace (["plymouth horizon miser"], 280)
dataset["name"] = dataset["name"].replace (["mazda glc custom"], 281)
dataset["name"] = dataset["name"].replace (["amc concord dl"], 282)
dataset["name"] = dataset["name"].replace (["chevrolet cavalier wagon"], 283)
dataset["name"] = dataset["name"].replace (["chrysler lebaron salon"], 284)
dataset["name"] = dataset["name"].replace (["volvo diesel"], 285)
dataset["name"] = dataset["name"].replace (["datsun 280-zx"], 286)
dataset["name"] = dataset["name"].replace (["audi 5000s (diesel)"], 287)
dataset["name"] = dataset["name"].replace (["peugeot 504 (sw)"], 288)
dataset["name"] = dataset["name"].replace (["datsun 310"], 289)
dataset["name"] = dataset["name"].replace (["toyota corolla tercel"], 290)
dataset["name"] = dataset["name"].replace (["fiat strada custom"], 291)
dataset["name"] = dataset["name"].replace (["plymouth horizon tc3"], 292)
dataset["name"] = dataset["name"].replace (["ford fairmont 4"], 293)
dataset["name"] = dataset["name"].replace (["buick regal sport coupe (turbo)"], 294)
dataset["name"] = dataset["name"].replace (["toyota celica gt liftback"], 295)
dataset["name"] = dataset["name"].replace (["audi 5000"], 296)
dataset["name"] = dataset["name"].replace (["saab 99gle"], 297)
dataset["name"] = dataset["name"].replace (["peugeot 604sl"], 298)
dataset["name"] = dataset["name"].replace (["honda accord lx"], 299)
dataset["name"] = dataset["name"].replace (["pontiac grand prix lj"], 300)
dataset["name"] = dataset["name"].replace (["honda accord cvcc"], 301)
dataset["name"] = dataset["name"].replace (["ford f108"], 302)
dataset["name"] = dataset["name"].replace (["capri ii"], 303)
dataset["name"] = dataset["name"].replace (["chevrolet monza 2+2"], 304)
dataset["name"] = dataset["name"].replace (["fiat x1.9"], 305)
dataset["name"] = dataset["name"].replace (["mercury monarch"], 306)
dataset["name"] = dataset["name"].replace (["dodge coronet custom"], 307)
dataset["name"] = dataset["name"].replace (["pontiac safari (sw)"], 306)
dataset["name"] = dataset["name"].replace (["ford f250"], 308)
dataset["name"] = dataset["name"].replace (["chevy c20"], 309)

CarNP = dataset.as_matrix()

carTargets =[]
for x in CarNP:
    carTargets.append(x[8])
CarNP = np.delete(CarNP, 8, 1)

dataNP = CarNP
targetNP = np.array(carTargets)

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