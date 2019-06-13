# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:31:00 2019

@author: kghiy
"""
import pandas as pd

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

X_train = train_set.iloc[:, train_set.columns != 'label']
y_train = train_set['label']

#import matplotlib
#import matplotlib.pyplot as plt

#some_digit = X_train.iloc[32000]
#some_digit = some_digit.values.reshape(28, 28)
#plt.imshow(some_digit, cmap = matplotlib.cm.binary, interpolation="nearest")
#plt.axis("off")
#plt.show()

#from sklearn.linear_model import SGDClassifier

#clf = SGDClassifier(random_state=42)
#clf.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42, n_estimators=1000)
clf.fit(X_train, y_train)

#from xgboost import XGBClassifier
#
#clf = XGBClassifier(n_estimators=100)
#clf.fit(X_train, y_train, verbose=False)

y_train_pred = clf.predict(train_set.iloc[:, train_set.columns != 'label'])

from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train, y_train_pred, average=None))
print(recall_score(y_train, y_train_pred, average=None))

import numpy as np

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

y_test = clf.predict(test_set)

submission = pd.DataFrame({
        "ImageId": test_set.index+1,
        "Label": y_test
    })
submission.to_csv('submission.csv', index=False)