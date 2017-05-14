# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:04:07 2017

@author: Alessandro
"""
#%%
random_state = 666

#%%
import pandas as pd
dataset = pd.read_csv("..\\common\\dataset.csv")

#%%
dataset = dataset.drop(["CUST_COD"], 1)
dataset["SEX"] = dataset["SEX"].astype('category').cat.codes
dataset["EDUCATION"] = dataset["EDUCATION"].astype('category').cat.codes
dataset["MARRIAGE"] = dataset["MARRIAGE"].astype('category').cat.codes
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
# Only AGE has NAN!!!
# Dirty trick
dataset['AGE'] = dataset['AGE'].fillna((dataset['AGE'].mean()))

#%%
# from sklearn.manifold import TSNE
# X_tsne = TSNE(learning_rate=100, n_iter=200, verbose=2).fit_transform(dataset)

#%%
target_col_name = "DEFAULT PAYMENT JAN"
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.33, random_state=random_state, stratify=dataset[target_col_name])

#%%
train[train["DEFAULT PAYMENT JAN"] == 1].shape[0] / train.shape[0]

#%%
test[test["DEFAULT PAYMENT JAN"] == 1].shape[0] / test.shape[0]  

#%%
dataset[dataset["DEFAULT PAYMENT JAN"] == 1].shape[0] / dataset.shape[0]

#%%
train_features = train.drop([target_col_name], 1)
train_target = train[target_col_name]
test_features = test.drop([target_col_name], 1)
test_target = test[target_col_name]

#%%
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
clf = tree.DecisionTreeClassifier(random_state=random_state, max_depth=3)
# clf = clf.fit(train_features, train_target)
from sklearn.model_selection import cross_val_score
# scores_nstkfold = cross_val_score(clf, train_features, train_target, cv=10)
scores_stkfold = cross_val_score(clf, train_features, train_target, scoring='f1_weighted', cv=StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True))
predicted_f1_test = scores_stkfold.mean()

#%%
clf = clf.fit(train_features, train_target)
from sklearn.metrics import f1_score
f1_train = f1_score(train_target, clf.predict(train_features), average='weighted')  
f1_test = f1_score(test_target, clf.predict(test_features), average='weighted')   
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(train_target, clf.predict(train_features))
cm_test = confusion_matrix(test_target, clf.predict(test_features))

#%%

target = pd.read_csv("./common/target.csv")
target = target.drop([target_col_name], 1)
target_nan = target[pd.isnull(target).any(axis=1)]
