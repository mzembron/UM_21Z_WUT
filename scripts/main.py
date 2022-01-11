import numpy as np
import pandas as pd
from missing_values_creator import MissingValuesCreator
import os
# path to data
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '..\\data\\iris.csv')

col_names = ["sepal.length","sepal.width","petal.length","petal.width","variety"]
data = pd.read_csv(filename, skiprows=1, header=None, names=col_names)
data.head(10)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

missing_values_creator = MissingValuesCreator()

X_test_missing = missing_values_creator.add_missing_values(X_test, 0)
print(X_test_missing)