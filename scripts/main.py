import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from missing_values_creator import MissingValuesCreator
from dt_missing_values import DecisionTreeClassifier
import os
# path to data
# path to data
dirname = os.path.dirname(__file__)
filename_iris = os.path.join(dirname, '../data/iris.csv')
filename_wine = os.path.join(dirname, '../data/wine.csv')

# data handling 

''' uncoment any data you are willing to use  '''

''' irises '''
# col_names_iris = ["sepal.length","sepal.width","petal.length","petal.width","variety"]
# data = pd.read_csv(filename_iris, skiprows=1, header=None, names=col_names_iris)

''' wines '''
col_names_wine = ["type","alcohol","malic.acid","ash","alcalinity.of.ash","magnesium","total.phenols","flavanoids","nonflavanoid.phenols","proanthocyanins","color.intensity","hue","OD280/OD315.of.diluted.wines","proline"]
data = pd.read_csv(filename_wine, skiprows=1, header=None, sep = ';',names=col_names_wine)
data = data[["alcohol","malic.acid","ash","alcalinity.of.ash","magnesium","total.phenols","flavanoids","nonflavanoid.phenols","proanthocyanins","color.intensity","hue","OD280/OD315.of.diluted.wines","proline","type"]]
# print(data)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

missing_values_creator = MissingValuesCreator()
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=np.inf) #there is no constraint on decision tree depth
classifier.fit(X_train,Y_train)
classifier.print_tree()

classifier1 = DecisionTreeClassifier(min_samples_split=3, max_depth=np.inf, missing_values_predictor= 2) #naive probabilistic approach to missing values 
classifier1.fit(X_train,Y_train)
classifier1.print_tree()


missing_values_creator = MissingValuesCreator(20)

X_test_missing =  missing_values_creator.add_missing_values(missing_values_creator.add_missing_values(missing_values_creator.add_missing_values(missing_values_creator.add_missing_values(X_test, 9),6),11),2)

Y_pred = classifier.predict(X_test_missing) 
Y_pred1 = classifier1.predict(X_test_missing) 

print(accuracy_score(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred1))