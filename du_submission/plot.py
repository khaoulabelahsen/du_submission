#!/usr/bin/python

import pandas as pd 
import plotly.express as px 

def plot_missing_values(dataset, type): 
    percent_missing = pd.DataFrame(dataset.isnull().sum() * 100 / len(dataset)).reset_index().rename(columns={'index': 'feature', 0:'percentage'}).sort_values(by='percentage')

    fig = px.bar(percent_missing, y='percentage', x='feature', orientation='v', title= 'Percentage of missing values in {type} set '.format(type=type))
    fig.show()

    return 0 