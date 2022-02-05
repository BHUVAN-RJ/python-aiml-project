import pandas as pd
import numpy as ny

datasets_train=pd.read_csv("/Users/bhuvanrj/Desktop/data science/train.csv")
datasets_test=pd.read_csv("/Users/bhuvanrj/Desktop/data science/test.csv")



X_train=datasets_train.iloc[:,:-1]
X_test=datasets_train.iloc[:,:-1]
Y_train=datasets_train.iloc[:,11]
Y_test=datasets_train.iloc[:,11]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A= make_column_transformer((OneHotEncoder(categories="auto"),[0]),remainder="passthrough")
X_train=A.fit_transform(X_train)
X_test=A.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(Y_test,y_pred))
print("model score",model.score(X_test,Y_test))




from matplotlib import pyplot as pp
pp.scatter(Y_test,y_pred)
pp.show()

#accuracy_score and confusion maatrix can only be implemented for classification not regression



