import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, f1_score, roc_curve, auc
from SequentialCoveringClassifier import SequentialCoveringClassifier

r = 666
np.random.seed(r)

dataset = pd.read_csv("../common/dataset.csv", header=0, index_col='CUST_COD')
# Backup original dataset
orig_dataset = dataset.copy()

print(orig_dataset.shape)

X_dict = {}
y_dict = {}
X_test_dict = {}
y_test_dict = {}

names = set()
target_col_name = 'DEFAULT PAYMENT JAN'

def split_and_add(dataset, name):
    train, test = train_test_split(dataset, test_size=0.33, random_state=r, stratify=dataset[target_col_name])
    X = train.drop([target_col_name], 1)
    y = train[target_col_name]
    X_test = test.drop([target_col_name], 1)
    y_test = test[target_col_name]
    add_dataset(X, y, X_test, y_test, name)
    print('Train X shape: {}'.format(X.shape))
    print('Test X shape: {}'.format(X_test.shape))

def add_dataset(X, y, X_test, y_test, name):
    names.add(name)
    X_dict[name] = X
    y_dict[name] = y
    X_test_dict[name] = X_test
    y_test_dict[name] = y_test

def retrieve_dataset(name):
    return (X_dict[name], y_dict[name], X_test_dict[name], y_test_dict[name])


orig_dataset = orig_dataset.drop(['SEX', 'EDUCATION', 'MARRIAGE', 'BIRTH_DATE'], 1)

#orig_dataset['PAY_DEC_MINUS_NOV'] = orig_dataset['PAY_DEC'] - orig_dataset['PAY_NOV']
#orig_dataset['PAY_AMT_DEC_OVER_LIMIT_BAL'] = orig_dataset['PAY_AMT_DEC'] / orig_dataset['LIMIT_BAL']
#orig_dataset['PAY_AMT_DEC_MINUS_LIMIT_BAL'] = orig_dataset['PAY_AMT_DEC'] - orig_dataset['LIMIT_BAL']
#orig_dataset['PAY_AMT_DEC_MINUS_PAY_AMT_NOV'] = orig_dataset['PAY_AMT_DEC'] - orig_dataset['PAY_AMT_NOV']
#orig_dataset['PIPPO2'] = orig_dataset['PAY_DEC'] * orig_dataset['PAY_AMT_DEC']
#orig_dataset['PIPPO'] = orig_dataset['PAY_AMT_DEC'] - orig_dataset['BILL_AMT_NOV']

split_and_add(dataset=orig_dataset.drop([
    #'SEX', 'EDUCATION', 'MARRIAGE', 'BIRTH_DATE',
    'BILL_AMT_DEC',
    'BILL_AMT_NOV',
    'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL',
    #'PAY_DEC',
    'PAY_NOV',
    'PAY_OCT',
    'PAY_SEP', 'PAY_AUG',
    'PAY_JUL',
    'PAY_AMT_DEC',
    'LIMIT_BAL',
    'PAY_AMT_NOV', 
    'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL'
    ], 1), name='baseline')

clf = SequentialCoveringClassifier()
X, y, X_test, y_test = retrieve_dataset('baseline')
# print(y)
#import sys
#import os
#sys.stdout = open(os.devnull, "w")
clf.fit(X, y)

def bad_good(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('{:.2f}% BAD customer predicted as GOOD customer'.format(cm[1][0] / (cm[1][0]+cm[1][1])))
    print('{:.2f}% GOOD customer predicted as BAD customer'.format(cm[0][1] / (cm[0][0]+cm[0][1])))

y_pred = clf.predict(X)
#sys.stdout = sys.__stdout__
bad_good(y, y_pred)
print(accuracy_score(y, y_pred))
print(f1_score(y, y_pred))
print(confusion_matrix(y, y_pred))

y_pred = clf.predict(X_test)
#sys.stdout = sys.__stdout__
bad_good(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
