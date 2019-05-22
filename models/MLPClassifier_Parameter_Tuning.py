import dill

import pandas as pd
import numpy as np

from sklearn import metrics

from sklearn.neural_network import MLPClassifier

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

X_cols = ['Sky_Group', 'i_period', 'i_epoch', 'N_Transit', 'i_depth', 'i_dur', 'i_b', 'i_ror', 'i_dor', 'Expected_MES']
X = temp_df[X_cols]
Y = temp_df.iloc[:, 14]

print('Input Size:', X.shape)
print('Output Size:', Y.shape)
print()

# Setting up the k-fold
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 7)

# Instantiate the model
mlp = MLPClassifier()

# Updating the param grid
param_grid = dict(activation = ['identity', 'logistic', 'tanh', 'relu'],
                  alpha = [0.0005, 0.0001, 0.005, 0.001, 0.01, 0.1],
                  batch_size = [32, 64, 96, 128, 256],
                  hidden_layer_sizes = [(50, 100, 50), (50, 100), (100, 50)],
                  learning_rate = ['constant', 'invscaling', 'adaptive'],
                  learning_rate_init = [0.0001, 0.001, 0.01, 0.1],
                  max_iter = [250, 500, 1000],
                  solver = ['lbfgs', 'sgd', 'adam'])

print('Multi-layer Perceptron Classifier')
mlp_grid_search = GridSearchCV(mlp, param_grid, scoring = 'accuracy', n_jobs = -1, cv = kfold, verbose = 1)
mlp_grid_result = mlp_grid_search.fit(X, Y)
mlp_predict = mlp_grid_search.predict(X)
mlp_predict_proba = pd.DataFrame(mlp_grid_search.predict_proba(X))

# Store metrics
mlp_accuracy = metrics.accuracy_score(Y, mlp_predict)  
mlp_precision = metrics.precision_score(Y, mlp_predict, pos_label=1)
mlp_recall = metrics.recall_score(Y, mlp_predict, pos_label=1)  
mlp_f1 = metrics.f1_score(Y, mlp_predict, pos_label=1)
mlp_auroc = metrics.roc_auc_score(Y, mlp_predict_proba[1])
mlp_aurpc = metrics.average_precision_score(Y, mlp_predict, pos_label=1)

dill.dump_session('MLPClassifier_Parameter_Tuning.db')

