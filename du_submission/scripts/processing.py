#!/usr/bin/python

import pandas as pd 
from collections import Counter 
import numpy as np 

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
        dataset[feature] = dataset[feature].apply(lambda x : float(str(x).strip('$').replace(',', '')))

    return 0 

def binary_processing(dataset):
    """Function to process specific features in the dataset 
    """
    # dataset['MSTATUS'] = dataset['MSTATUS'].apply(lambda x : 1 if x =='Yes' else 0)
    # dataset['SEX'] = dataset['SEX'].apply(lambda x : 1 if x =='M' else 0)
    # dataset['PARENT1'] = dataset['PARENT1'].apply(lambda x : 1 if x =='Yes' else 0)
    # dataset['RED_CAR'] = dataset['RED_CAR'].apply(lambda x : 1 if x =='yes' else 0)
    # dataset['REVOKED'] = dataset['REVOKED'].apply(lambda x : 1 if x =='Yes' else 0)

    dataset['MSTATUS'] = dataset['MSTATUS'].astype('category')
    dataset['SEX'] = dataset['SEX'].astype('category')
    dataset['PARENT1'] = dataset['PARENT1'].astype('category')
    dataset['RED_CAR'] = dataset['RED_CAR'].astype('category')
    dataset['REVOKED'] = dataset['REVOKED'].astype('category')

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

def detect_outlier(df):
    features = df.columns
    outliers  = []
    for i, feature in enumerate(features):
        if df[feature].dtype == 'float64':
            # Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(df[feature], 25)
            # Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(df[feature], 75)
            # Use the interquartile range to calculate an outlier step
            step = 1.5 * (Q3 - Q1)
            feature_outliers = df[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step))]
            outliers.extend(list(feature_outliers.index.values))
            print('Feature: {}, outliers: {}\n'.format(feature, len(feature_outliers.index)))
    
    features_outliers = (Counter(outliers) - Counter(set(outliers))).keys()
    return features_outliers
