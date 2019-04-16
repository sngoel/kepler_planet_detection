import dill

import pandas as pd
import numpy as np

from sklearn import metrics

from sklearn.ensemble import ExtraTreesClassifier

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
etc = ExtraTreesClassifier()

# Updating the param grid
param_grid = dict(bootstrap = ['True', 'False'],
                  criterion = ['gini', 'entropy'],
                  max_depth = [int(x) for x in np.linspace(start = 2, stop = 30, num = 10)],
                  max_features = ['auto', 'sqrt', 'log2', None],
                  min_impurity_decrease = [0.00001, 0.0001, 0.001, 0.01, 0.1],
                  min_samples_split = [2, 4, 6, 8, 10],
                  min_samples_leaf = [0.10, 0.25, 0.50, 1, 2, 4],
                  n_estimators = [int(x) for x in np.linspace(start = 10, stop = 350, num = 10)])

print('ExtraTrees Classifier')
etc_grid_search = GridSearchCV(etc, param_grid, scoring = 'accuracy', n_jobs = -1, cv = kfold, verbose = 1)
etc_grid_result = etc_grid_search.fit(X, Y)
etc_predict = etc_grid_search.predict(X)
etc_predict_proba = pd.DataFrame(etc_grid_search.predict_proba(X))

# Store metrics
etc_accuracy = metrics.accuracy_score(Y, etc_predict)  
etc_precision = metrics.precision_score(Y, etc_predict, pos_label=1)
etc_recall = metrics.recall_score(Y, etc_predict, pos_label=1)  
etc_f1 = metrics.f1_score(Y, etc_predict, pos_label=1)
etc_auroc = metrics.roc_auc_score(Y, etc_predict_proba[1])
etc_aurpc = metrics.average_precision_score(Y, etc_predict, pos_label=1)

dill.dump_session('ExtraTreesClassifier_Parameter_Tuning.db')