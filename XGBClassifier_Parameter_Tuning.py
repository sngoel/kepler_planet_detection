import dill

import pandas as pd
import numpy as np

from sklearn import metrics

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Setting a random seed for reproducability
np.random.seed(7)

# Problem 1
df = pd.read_csv('kplr_dr25_inj1_plti.csv', header = 0)

print('Dataset Size:')
print(df.shape)
print()

temp_df = df.iloc[:, 0:15]
df_drop = temp_df[temp_df.isnull().any(axis=1)]
temp_df = temp_df.drop(df_drop.index.values)
temp_df = temp_df[temp_df.Recovered != 2]

print('Cleaned Dataset Size:')
print(temp_df.shape)
print()

X = temp_df.iloc[:, 1:14]
Y = temp_df.iloc[:, 14]

print('Input Size:', X.shape)
print('Output Size:', Y.shape)
print()

# Setting up the k-fold
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 7)

# Instantiate the model
xgb = XGBClassifier()

# Updating the param grid
param_grid = dict(booster = ['gbtree', 'gblinear', 'dart'],
                  learning_rate = [0.0001, 0.001, 0.01, 0.1, 1.0],
                  min_split_loss = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                  max_depth = [int(x) for x in np.linspace(start = 2, stop = 30, num = 10)],
                  min_child_weight = [1, 3, 5],
                  reg_lambda = [0.0001, 0.001, 0.01, 0.1, 0.0, 1.0],
                  reg_alpha = [0.0001, 0.001, 0.01, 0.1, 0.0, 1.0],
                  tree_method = ['exact', 'approx', 'hist'])

print('XGBoost Classifier')
xgb_grid_search = GridSearchCV(xgb, param_grid, scoring = 'accuracy', n_jobs = -1, cv = kfold, verbose = 1)
xgb_grid_result = xgb_grid_search.fit(X, Y)
xgb_predict = xgb_grid_search.predict(X)
xgb_predict_proba = pd.DataFrame(xgb_grid_search.predict_proba(X))

# Store metrics
xgb_accuracy = metrics.accuracy_score(Y, xgb_predict)  
xgb_precision = metrics.precision_score(Y, xgb_predict, pos_label=1)
xgb_recall = metrics.recall_score(Y, xgb_predict, pos_label=1)  
xgb_f1 = metrics.f1_score(Y, xgb_predict, pos_label=1)
xgb_auroc = metrics.roc_auc_score(Y, xgb_predict_proba[1])
xgb_aurpc = metrics.average_precision_score(Y, xgb_predict, pos_label=1)

dill.dump_session('XGBClassifier_Parameter_Tuning.db')