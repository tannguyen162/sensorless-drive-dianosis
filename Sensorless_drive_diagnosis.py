#doc
import pandas as pd
dataset=pd.read_csv('Sensorless_drive_diagnosis.txt', sep=" ", header=None)

# kiem tra co rong khong
dataset.isnull().values.any()

#gan data, target
data = dataset.iloc[:,0:48]
target = dataset.iloc[:,48:49]

#phan chia du lieu
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=1/5, shuffle=True, random_state = 42)

#xay dung mo hinh
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


#du doan
y_pred=clf.predict(X_test)
y_pred

#danh gia
from sklearn.metrics import accuracy_score
print("DO CHINH XAC: ")
accuracy_score(y_test,y_pred)*100


#========bieu do xep hang thuoc tinh==========
#from matplotlib import pyplot as plt
#import numpy as np
#importances=clf.feature_importances_

#indices = np.argsort(importances)[::-1]

# xep hang thuoc tinh
#print("xep han thuoc tinh:")

#for f in range(X_train.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

#plt.figure(figsize=(15,4))
#plt.title("Feature importances")
#plt.bar(range(X_train.shape[1]), importances[indices],color="b", align="center")
#plt.xticks(range(X_train.shape[1]), indices)
#plt.xlim([-1, X_train.shape[1]])
#plt.show()