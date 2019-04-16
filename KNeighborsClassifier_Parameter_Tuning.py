import dill

import pandas as pd
import numpy as np

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

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
knc = KNeighborsClassifier()

# Updating the param grid
param_grid = dict(algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  leaf_size = [10, 20, 30, 40, 50],
                  metric = ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                  n_neighbors = [int(x) for x in np.linspace(start = 3, stop = 30, num = 10)],
                  p = [float(x) for x in np.linspace(start = 1, stop = 5, num = 10)],
                  weights = ['uniform', 'distance'])

print('KNeighbors Classifier')
knc_grid_search = GridSearchCV(knc, param_grid, scoring = 'accuracy', n_jobs = -1, cv = kfold, verbose = 1)
knc_grid_result = knc_grid_search.fit(X, Y)
knc_predict = knc_grid_search.predict(X)
knc_predict_proba = pd.DataFrame(knc_grid_search.predict_proba(X))

# Store metrics
knc_accuracy = metrics.accuracy_score(Y, knc_predict)  
knc_precision = metrics.precision_score(Y, knc_predict, pos_label=1)
knc_recall = metrics.recall_score(Y, knc_predict, pos_label=1)  
knc_f1 = metrics.f1_score(Y, knc_predict, pos_label=1)
knc_auroc = metrics.roc_auc_score(Y, knc_predict_proba[1])
knc_aurpc = metrics.average_precision_score(Y, knc_predict, pos_label=1)

dill.dump_session('KNeighborsClassifier_Parameter_Tuning.db')
