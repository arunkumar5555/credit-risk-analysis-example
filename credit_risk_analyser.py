
import pandas as pd

import numpy as np

import random as rnd

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split

# from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt



df = pd.read_csv(r"C:\Users\lenovo\Desktop\work\credit risk analysys\credit_data.csv")

print(df)

print(df.describe())





df1 = pd.get_dummies(df['gender'],drop_first=True)

df2 = pd.get_dummies(df['education'],drop_first=True)

df3 = pd.get_dummies(df['occupation'],drop_first=True)

df4 = pd.get_dummies(df['organization_type'],drop_first=True)

df5 = pd.get_dummies(df['seniority'],drop_first=True)

df6 = pd.get_dummies(df['house_type'],drop_first=True)

df7 = pd.get_dummies(df['vehicle_type'],drop_first=True)

df8 = pd.get_dummies(df['marital_status'],drop_first=True)

df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df],axis=1)

print(df)



df.drop(['gender','education','occupation','organization_type','seniority','house_type','vehicle_type','marital_status'],axis=1,inplace=True)

print(df.shape)



array = df.values

X = array[:,0:26]

Y = array[:,26]

print(array.dtype)

print(X,Y)





X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)



model = DecisionTreeClassifier()



model.fit(X_train, Y_train)



DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

            max_features=None, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, presort=False, random_state=None,

            splitter='best')



rnd.seed(123458)

X_new = X[rnd.randrange(X.shape[0])]

X_new = X_new.reshape(1,26)

YHat = model.predict(X_new)





df9 = pd.DataFrame(X_new)

df9["predicted"] = YHat

df9.head()

print(df9)



YHat = model.predict(X_test)



# calculate accuracy

print (" gini accuracy is : ",round(accuracy_score(Y_test, YHat)*100,2))



model = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes = 20)

dt = model.fit(X_train, Y_train)



YHat = model.predict(X_test)

print (" entropy accuracy is : " ,round(accuracy_score(Y_test, YHat)*100,2))
