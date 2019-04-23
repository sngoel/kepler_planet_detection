import dill

import pandas as pd
import numpy as np

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Setting a random seed for reproducability
np.random.seed(7)

# Problem 2
df = pd.read_csv('kplr_dr25_inj1_tces.csv', header = 0)

print('Dataset Size: ')
print(df.shape)
print()

cols = ['TCE_ID', 'KIC', 'Disp', 'Score', 'period', 'epoch', 'NTL', 'SS', 'CO', 'EM', 'Expected_MES', 'MES', 'NTran',
        'depth', 'duration', 'Rp', 'Rs', 'Ts', 'logg', 'a', 'Rp/Rs', 'a/Rs', 'impact', 'SNR_DV', 'Sp', 'Fit_Prov']
df = df[cols]
df.columns

df['Disp'] = df['Disp'].replace('PC', 1)
df['Disp'] = df['Disp'].replace('FP', 0)

X = df.iloc[:, 10:]
Y = df.iloc[:, 2]

print('Input Size:', X.shape)
print('Output Size:', Y.shape)
print()

# Setting up the k-fold
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 7)

# Instantiate the model
rfc = RandomForestClassifier()

# Updating the param grid
param_grid = dict(max_depth = [int(x) for x in np.linspace(start = 2, stop = 30, num = 5)],
                  max_features = ['sqrt', 'log2', None],
                  min_impurity_decrease = [0.00001, 0.0001, 0.001, 0.01, 0.1],
                  min_samples_split = [2, 4, 6, 8, 10],
                  min_samples_leaf = [0.10, 0.25, 0.50, 1, 2, 4],
                  n_estimators = [int(x) for x in np.linspace(start = 10, stop = 350, num = 5)])

print('RandomForest Classifier')
rfc_grid_search = GridSearchCV(rfc, param_grid, scoring = 'accuracy', n_jobs = -1, cv = kfold, verbose = 1)
rfc_grid_result = rfc_grid_search.fit(X, Y)
rfc_predict = rfc_grid_search.predict(X)
rfc_predict_proba = pd.DataFrame(rfc_grid_search.predict_proba(X))

# Store metrics
rfc_accuracy = metrics.accuracy_score(Y, rfc_predict)  
rfc_precision = metrics.precision_score(Y, rfc_predict, pos_label=1)
rfc_recall = metrics.recall_score(Y, rfc_predict, pos_label=1)  
rfc_f1 = metrics.f1_score(Y, rfc_predict, pos_label=1)
rfc_auroc = metrics.roc_auc_score(Y, rfc_predict_proba[1])
rfc_aurpc = metrics.average_precision_score(Y, rfc_predict, pos_label=1)

dill.dump_session('RandomForestClassifier_Parameter_Tuning.db')