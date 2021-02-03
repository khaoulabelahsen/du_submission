#!/usr/bin/python

import pandas as pd 

def full_processing(dataset):
    """Return the full processed version of any dataset with the same features as the train set 
    """
    money_processing(dataset, ['INCOME', 'HOME_VAL','BLUEBOOK', 'OLDCLAIM'] )
    binary_processing(dataset)
    dataset2 = get_dummy(dataset, ['URBANICITY', 'CAR_TYPE', 'CAR_USE', 'JOB', 'EDUCATION'])    
    return dataset2 

def money_processing(dataset, features):
    """Returns the new dataset with processed features 

    Inputs : 
        - dataset 
        - features : list of features to process
    """
    for feature in features: 
        dataset[feature] = dataset[feature].apply(lambda x : float(str(x).strip('$').replace(',', '.')))

    return 0 

def binary_processing(dataset):
    """Function to process specific features in the dataset 
    """
    dataset['MSTATUS'] = dataset['MSTATUS'].apply(lambda x : 1 if x =='Yes' else 0)
    dataset['SEX'] = dataset['SEX'].apply(lambda x : 1 if x =='M' else 0)
    dataset['PARENT1'] = dataset['PARENT1'].apply(lambda x : 1 if x =='Yes' else 0)
    dataset['RED_CAR'] = dataset['RED_CAR'].apply(lambda x : 1 if x =='yes' else 0)
    dataset['REVOKED'] = dataset['REVOKED'].apply(lambda x : 1 if x =='Yes' else 0)

    return 0 

def get_dummy(dataset, features):
    """Returns the new dataset with dummy features 

    Inputs : 
        - dataset 
        - features : list of features to dummify
    """

    for feature in features: 
        y = pd.get_dummies(dataset[feature])
        #dataset = dataset.drop(feature,axis = 1)
        # Join the encoded df
        dataset = dataset.join(y)

    return dataset


def replace_with_method(dataset, features, method):
    """Fills the missing values in the dataset's features following method 
    """

    for feature in features:

        if method == 'mean':
            dataset[feature]= dataset[[feature]].fillna(dataset[feature].mean())
        elif method == 'median':
            dataset[feature]= dataset[[feature]].fillna(dataset[feature].median())
        else : 
            raise NotImplementedError

    return 0 
