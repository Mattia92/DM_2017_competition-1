# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:04:07 2017

@author: Alessandro
"""
#%%
random_state = 666

import pandas as pd
dataset = pd.read_csv("..\common\dataset.csv")

fields = {'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_DEC', 'PAY_NOV', 'PAY_OCT', 'PAY_SEP', 'PAY_AUG', 'PAY_JUL'}
for field in fields:
    dataset = pd.get_dummies(dataset[field], prefix=field).join(dataset.drop([field], 1))
    
# set(dataset["SEX"]) = {nan, 'M', 'F'}
# set(dataset["EDUCATION"]) = {nan, 'high school', 'graduate school', 'other', 'university'}
# set(dataset["MARRIAGE"]) = {nan, 'other', 'single', 'married'}

# def edu_to_val(edu):
#    if edu == None:
#        return 0
#    elif edu == "other":
#        return 1
#    elif edu == "high school":
#        return 2
#    elif edu == "graduate school":
#        return 3
#    elif edu == "university":
#        return 4
#    else:
#        return 0
#dataset['EDUCATION'] = dataset['EDUCATION'].apply(lambda x: edu_to_val(x))

#def marriage_to_val(marriage):
#    if marriage == None:
#        return 0
#    elif marriage == "other":
#        return 1
#    elif marriage == "single":
#        return 2
#    elif marriage == "married":
#        return 3
#    else:
#        return 0
#dataset["MARRIAGE"] = dataset["MARRIAGE"].apply(lambda x: marriage_to_val(x))

dataset['BIRTH_DATE'] = pd.to_datetime(dataset['BIRTH_DATE'], format='%d/%m/%Y')
from datetime import date
def calculate_age(birth_date):
    today = date.today()
    age = today.year - birth_date.year
    full_year_passed = (today.month, today.day) < (birth_date.month, birth_date.day)
    if not full_year_passed:
        age -= 1
    return age
dataset['AGE'] = dataset['BIRTH_DATE'].apply(lambda x: calculate_age(x))
dataset = dataset.drop(["BIRTH_DATE"], 1)
dataset['AGE'] = dataset['AGE'].fillna(dataset['AGE'].mean())

from sklearn import preprocessing

fields = {'LIMIT_BAL', 'BILL_AMT_DEC', 'BILL_AMT_NOV', 'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'PAY_AMT_DEC', 'PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL', 'AGE'}
for field in fields:
    dataset[field] = preprocessing.scale(dataset[field])

dataset = dataset.drop(['CUST_COD'], 1)

#%%
# from sklearn.manifold import TSNE
# X_tsne = TSNE(learning_rate=100, n_iter=200, verbose=2).fit_transform(dataset)

#%%
target_col_name = "DEFAULT PAYMENT JAN"
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.33, random_state=random_state, stratify=dataset[target_col_name])
# print(train[train["DEFAULT PAYMENT JAN"] == 1].shape[0] / train.shape[0])
# print(test[test["DEFAULT PAYMENT JAN"] == 1].shape[0] / test.shape[0])
# print(dataset[dataset["DEFAULT PAYMENT JAN"] == 1].shape[0] / dataset.shape[0])
train_features = train.drop([target_col_name], 1)
y = train[target_col_name].tolist()
test_features = test.drop([target_col_name], 1)
y_test = test[target_col_name].tolist()

#%%
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(train_features)
cumsum = pca.explained_variance_ratio_.cumsum()

cumsum = sorted(list(cumsum))
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(random_state)
plt.figure(1)
plt.plot(cumsum, label='Cumsum')
plt.xlabel('N Components')
plt.ylabel('Cumsum')
plt.title('PCA Cumsum')
plt.legend(loc='best')
plt.show()
cumsum_tresh = 0.95
for i in np.arange(0,len(cumsum)):
    if cumsum[i] >= cumsum_tresh:
        break
n_components = i
print('cumsum_tresh = {}, n_components = {}'.format(cumsum_tresh, n_components))
pca = PCA(n_components=n_components)
pca.fit(train_features)
X = pca.transform(train_features)
X_test = pca.transform(test_features)

#%%
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)

#%%
def test_clf(X, y, y_pred, X_test, y_test, y_pred_test):
    from sklearn.metrics import f1_score
    f1_train = f1_score(y, y_pred, average='macro')  
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    print('f1_train = \n\t{}\nf1_test = \n\t{}'.format(f1_train, f1_test))
    from sklearn.metrics import confusion_matrix
    cm_train = confusion_matrix(y, y_pred)
    cm_test = confusion_matrix(y_test, y_pred_test)
    print('cm_train = \n{}\ncm_test = \n{}'.format(cm_train, cm_test))
    cost_train = cost_matrix(y, y_pred)
    cost_test = cost_matrix(y_test, y_pred_test)
    print('cost_train = \n{}\ncost_test = \n{}'.format(format(cost_train, ','), format(cost_test, ',')))
    
#%%
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(random_state=random_state)
clf.fit(X, y)
test_clf(X, y, clf.predict(X), X_test, y_test, clf.predict(X_test))

#%%
from sklearn import svm
clf = svm.SVC(random_state=random_state, verbose=2)
clf.fit(X, y)
test_clf(X, y, clf.predict(X), X_test, y_test, clf.predict(X_test))

#%%
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=random_state)
clf.fit(X, y)
test_clf(X, y, clf.predict(X), X_test, y_test, clf.predict(X_test))

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix

def cost_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    # first index true
    # second index predicted
    return (cm[0][0] * 0 + cm[0][1] * 1)*0.5/0.8 + (cm[1][0] * 1 + cm[1][1] * 0)*0.5/0.2
ms = make_scorer(cost_matrix, greater_is_better=False)

# parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
# {'C': 100}
cv=StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)
parameters = {'C': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10e4, 10e5, 10e6, 10e7]}
#parameters = {'class_weight' : {0:5/8, 1:5/2}}
#parameters = {'C' : np.arange(1, 150, step=1)} 
            # 'fit_intercept': [True, False],
             # 'penalty': ['l1', 'l2']}
#parameters = {'C': [1e-1, 1, 10]}
clf = LogisticRegression(penalty='l1', C=1e10,  fit_intercept=True, random_state=random_state, verbose=0, n_jobs=1)
gscv = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=-1, cv=cv, scoring='roc_auc')
gscv.fit(X, y)
plt.figure(1)
#plt.plot(parameters['C'], gscv.cv_results_['mean_test_score'], label='Score')
plt.semilogx(parameters['C'], gscv.cv_results_['mean_test_score'], label='Score')
plt.xlabel('C')
plt.ylabel('Score')
plt.title('CV Score')
plt.legend(loc='best')
plt.show()
print(gscv.best_params_)

#%%
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1', C=1, class_weight=None, fit_intercept=True, random_state=random_state, verbose=2, n_jobs=-1)
clf.fit(X, y)
test_clf(X, y, clf.predict(X), X_test, y_test, clf.predict(X_test))

#%%
# clf = clf.fit(train_features, train_target)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)
# scores_nstkfold = cross_val_score(clf, train_features, train_target, cv=10)
scores_stkfold = cross_val_score(clf, train_features, train_target, scoring='f1_macro', cv=cv)
predicted_f1_test = scores_stkfold.mean()

#%%

target = pd.read_csv("./common/target.csv")
target = target.drop([target_col_name], 1)
target_nan = target[pd.isnull(target).any(axis=1)]
