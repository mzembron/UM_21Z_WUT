import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from missing_values_creator import MissingValuesCreator
from dt_missing_values import DecisionTreeClassifier
import os

def test_percentage_of_missing_values_in_data():
    # loading data
    dirname = os.path.dirname(__file__)
    filename_iris = os.path.join(dirname, '../data/iris.csv')
    data = pd.read_csv(filename_iris, skiprows=1, header=None)
    X = data.iloc[:, :-1].values

    percentage = 65
    index = 1
    
    missing_values_creator = MissingValuesCreator(percentage)
    X_missing_values = missing_values_creator.add_missing_values_by_list(X, [index])
    number_of_elements_missing = 0
    
    # it was easier to loop than use count() method - "nan" has some stange type
    for element in X_missing_values[:, 1].tolist():
        if str(element) == "nan":
            number_of_elements_missing += 1

    number_of_all_elements = len(X_missing_values[:, 1].tolist())
    assert(round(number_of_elements_missing/number_of_all_elements,2) == (percentage/100))

    



