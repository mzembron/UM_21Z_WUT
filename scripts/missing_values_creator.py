import numpy as np
import pandas as pd
import random

class MissingValuesCreator:
    
    def __init__(self, percent = 10):
        self.percent = percent

    def change_percent(self, percent):
        self.percent = percent

    def add_missing_values(self, list, index):
        '''function adding missing values to given data at given index'''
        #index - index of parameter of element which will be missing

        list_with_missing_values = list
        indexes_of_not_missing_elements = np.linspace(0, len(list)-1, len(list))
        
        # loop until given percantge of missing values will be obtained
        while(len(indexes_of_not_missing_elements)>(len(list)*((100-self.percent)/100))):
            #randomly selecting element of list (element without missing values)
            index_of_missing_value = random.choice(indexes_of_not_missing_elements)
            #filling with missing value
            list_with_missing_values[int(index_of_missing_value)][index] =  'NaN'
            #deleting index from list of indexes of elements without missing values
            indices = np.where(indexes_of_not_missing_elements == index_of_missing_value)
            indexes_of_not_missing_elements = np.delete(indexes_of_not_missing_elements, indices)
            # indexes_of_not_missing_elements.remove(index_of_missing_value)

        return list_with_missing_values

    def add_missing_values_by_list(self, list, list_of_indexes):
        '''function adding missing values to given data at given index'''
        #list_of_indexes - list of indexes of parameters which will be missing

        list_with_missing_values = list
        
        for index in list_of_indexes:
            indexes_of_not_missing_elements = np.linspace(0, len(list)-1, len(list))
            
            # loop until given percantge of missing values will be obtained
            while(len(indexes_of_not_missing_elements)>(len(list)*((100-self.percent)/100))):
                #randomly selecting element of list (element without missing values)
                index_of_missing_value = random.choice(indexes_of_not_missing_elements)
                #filling with missing value
                list_with_missing_values[int(index_of_missing_value)][index] =  'NaN'
                #deleting index from list of indexes of elements without missing values
                indices = np.where(indexes_of_not_missing_elements == index_of_missing_value)
                indexes_of_not_missing_elements = np.delete(indexes_of_not_missing_elements, indices)
                # indexes_of_not_missing_elements.remove(index_of_missing_value)

        return list_with_missing_values
            

    