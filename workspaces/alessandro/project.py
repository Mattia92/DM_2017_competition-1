# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:04:07 2017

@author: Alessandro
"""
#%%
random_state = 666

import pandas as pd
dataset = pd.read_csv("../common/dataset.csv")

fields = {'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_DEC', 'PAY_NOV', 'PAY_OCT', 'PAY_SEP', 'PAY_AUG', 'PAY_JUL'}
for field in fields:
    dataset = pd.get_dummies(dataset[field], prefix=field).join(dataset.drop([field], 1))
    
# set(dataset["SEX"]) = {nan, 'M', 'F'}
# set(dataset["EDUCATION"]) = {nan, 'high school', 'graduate school', 'other', 'university'}
# set(dataset["MARRIAGE"]) = {nan, 'other', 'single', 'married'}

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

fields = ['LIMIT_BAL',
          'BILL_AMT_DEC', 'BILL_AMT_NOV', 'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL',
          'PAY_AMT_DEC', 'PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL']
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
rs.fit(dataset[fields])
dataset[fields] = rs.transform(dataset[fields])
rs.fit(dataset['AGE'])
dataset['AGE'] = rs.transform(dataset['AGE'])
    
dataset = dataset.drop(['CUST_COD'], 1)

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
pca = PCA(whiten=True)
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
cumsum_tresh = 0.99
for i in np.arange(0,len(cumsum)):
    if cumsum[i] >= cumsum_tresh:
        break
n_components = i
print('cumsum_tresh = {}, n_components = {}'.format(cumsum_tresh, n_components))
pca = PCA(whiten=True, n_components=n_components)
pca.fit(train_features)
X = pca.transform(train_features)
X_test = pca.transform(test_features)

pca = PCA(whiten=True, n_components=2)
colors = {0:'r', 1:'b'}
df = pd.DataFrame(pca.fit_transform(test_features))
fig, ax = plt.subplots()
ax.scatter(df[0], df[1], c=pd.DataFrame(y)[0].apply(lambda x: colors[x]), s=1)
plt.show()

#%%
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)
from sklearn.metrics import make_scorer, confusion_matrix

def cost_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    # first index true
    # second index predicted
    return cm[0][0] * 0 + cm[0][1] * 1 + cm[1][0] * 7 + cm[1][1] * 0
ms = make_scorer(cost_matrix, greater_is_better=False)

#%%
def test_clf(X, y, y_pred, X_test, y_test, y_pred_test):
    from sklearn.metrics import f1_score
    f1_train = f1_score(y, y_pred, average='macro')  
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    print('f1_train = {}, f1_test = {}'.format(f1_train, f1_test))
    cost_train = cost_matrix(y, y_pred)
    cost_test = cost_matrix(y_test, y_pred_test)
    print('cost_train = {}, cost_test = {}'.format(format(cost_train, ','), format(cost_test, ',')))
    from sklearn.metrics import confusion_matrix
    cm_train = confusion_matrix(y, y_pred)
    cm_test = confusion_matrix(y_test, y_pred_test)
    print('cm_train = \n{}\ncm_test = \n{}'.format(cm_train, cm_test))
    
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
cv=StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)

parameters = {'C': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1,
                    1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]}
clf = LogisticRegression(penalty='l2', class_weight='balanced', fit_intercept=True, random_state=random_state, n_jobs=1)

#parameters = {'max_depth':np.arange(1,20)}
#clf = tree.DecisionTreeClassifier(random_state=random_state)

#parameters = {'max_depth':np.arange(1,20)}
#clf = svm.SVC(random_state=random_state, verbose=2)
scoring = 'f1'
gscv = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=-1, cv=cv, scoring=ms)
gscv.fit(X, y)
plt.figure(1)
#plt.plot(parameters['C'], gscv.cv_results_['mean_test_score'], label='Score')
plt.semilogx(parameters['C'], gscv.cv_results_['mean_test_score'], label='f1')
#plt.plot(parameters['max_depth'], gscv.cv_results_['mean_test_score'], label=scoring)
plt.xlabel('C')
plt.ylabel(scoring)
plt.title('CV ' + scoring)
plt.legend(loc='best')
plt.show()
print(gscv.best_params_)

#%%
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', C=0.001, class_weight='balanced',
                         fit_intercept=True, random_state=random_state, n_jobs=-1)
clf.fit(X, y)
test_clf(X, y, clf.predict(X), X_test, y_test, clf.predict(X_test))

from sklearn.metrics import roc_curve, auc

y_score = clf.fit(X, y).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, t = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

t_sigm = 1 / (1 + np.exp(-t))

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

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
