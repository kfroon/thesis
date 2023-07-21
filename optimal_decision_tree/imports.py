import sklearn
import pandas as pd 
import numpy as np
import pydot
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
import six
import sys
from interpretableai import iai
from matplotlib import pyplot


def load_data(df, split = True):
    X = df.iloc[:,:-1]

    feature_names = X.columns
    if split == False:
        return X, y, feature_names
    if split == True:
        (x_train, y_train), (x_test, y_test) = iai.split_data('classification', X, y,seed = 1)
        return x_train, x_test, y_train, y_test, feature_names


def load_classifier(x_train, y_train, classifier,maxdepth = list, min_samples_leaf= list, ):
   classifier = classifier
   param_grid = {'max_depth': maxdepth,
                  'criterion' :['gini', 'entropy'],
                  'min_samples_leaf': min_samples_leaf}
   grid_search = sklearn.model_selection.GridSearchCV(estimator=classifier,  
                           param_grid = param_grid, 
                           scoring = ['accuracy', 'f1'],
                           refit = 'accuracy',
                           cv=5, 
                           verbose=True, 
                           return_train_score=True)
   grid_search.fit(x_train, y_train)
   model = grid_search
   return model
