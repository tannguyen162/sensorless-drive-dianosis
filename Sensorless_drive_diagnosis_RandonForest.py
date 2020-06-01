# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:25:19 2019

@author: rang pham
"""

import pandas as pd
dt = pd.read_csv("Sensorless_drive_diagnosis.txt", delim_whitespace=True, header=None)

#gan data, target
data = dt.iloc[:,0:48]
target = dt.iloc[:,48:49]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=42)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
#du doan
y_pred=clf.predict(X_test)
y_pred
#do chinh xac
from sklearn.metrics import accuracy_score
print("DO CHINH XAC: ")
ac = accuracy_score(y_test,y_pred)*100
print(ac)
