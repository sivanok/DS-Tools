# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:58:20 2019

@author: User
"""

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn import  metrics  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def Gradient_Boosting(gb,features_train,label_train):

    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    
    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf}

    gf_tune = GridSearchCV(estimator = gb, param_grid = grid, cv = 2, verbose=2)
    gf_tune.fit(features_train, label_train)
    GB_clf = GradientBoostingClassifier(**gf_tune.best_params_)
    GB_clf.fit(features_train,label_train)
    train_fpr, train_tpr, _ = metrics.roc_curve(np.array(label_train), GB_clf.predict_proba(features_train)[:,1])
    gb_auc = metrics.auc(train_fpr,train_tpr)

    return gb_auc



def Random_Forest(rf,features_train,label_train):

    param_grid = {
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 2)
    CV_rfc.fit(features_train,label_train)
    train_fpr, train_tpr, _ = metrics.roc_curve(np.array(label_train), CV_rfc.predict_proba(features_train)[:,1])
    rf_auc = metrics.auc(train_fpr,train_tpr)
    
    return rf_auc
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
