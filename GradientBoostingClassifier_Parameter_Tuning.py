import dill

import pandas as pd
import numpy as np

from sklearn import metrics

from sklearn.ensemble import GradientBoostingClassifier

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
gbc = GradientBoostingClassifier()
        
# Updating the param grid
param_grid = dict(criterion = ['friedman_mse', 'mse', 'mae'],
                  learning_rate = [0.0001, 0.001, 0.01, 0.1, 1.0],
                  loss = ['deviance', 'exponential'],
                  max_depth = [int(x) for x in np.linspace(start = 2, stop = 30, num = 10)],
                  max_features = ['auto', 'sqrt', 'log2', None],
                  min_impurity_decrease = [0.00001, 0.0001, 0.001, 0.01, 0.1],
                  min_samples_split = [2, 4, 6, 8, 10],
                  min_samples_leaf = [0.10, 0.25, 0.50, 1, 2, 4],
                  n_estimators = [int(x) for x in np.linspace(start = 10, stop = 350, num = 10)],
                  subsample = [0.5, 1.0])
        
print('GradientBoosting Classifier')
gbc_grid_search = GridSearchCV(gbc, param_grid, scoring = 'accuracy', n_jobs = -1, cv = kfold, verbose = 1)
gbc_grid_result = gbc_grid_search.fit(X, Y)
gbc_predict = gbc_grid_search.predict(X)
gbc_predict_proba = pd.DataFrame(gbc_grid_search.predict_proba(X))

# Store metrics
gbc_accuracy = metrics.accuracy_score(Y, gbc_predict)  
gbc_precision = metrics.precision_score(Y, gbc_predict, pos_label=1)
gbc_recall = metrics.recall_score(Y, gbc_predict, pos_label=1)  
gbc_f1 = metrics.f1_score(Y, gbc_predict, pos_label=1)
gbc_auroc = metrics.roc_auc_score(Y, gbc_predict_proba[1])
gbc_aurpc = metrics.average_precision_score(Y, gbc_predict, pos_label=1)

dill.dump_session('GradientBoostingClassifier_Parameter_Tuning.db')