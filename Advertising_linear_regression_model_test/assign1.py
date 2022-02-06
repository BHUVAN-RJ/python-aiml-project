import pandas as pd
import numpy as ny

datasets= pd.read_csv("/Users/bhuvanrj/Desktop/data science/Advertising.csv")//your file path name

X=datasets.iloc[:,:-1]
Y=datasets.iloc[:,3]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression
model =LinearRegression()
model.fit(X_train,Y_train)
y_pred= model.predict(X_test)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test,y_pred))
print(mean_squared_error(Y_test,y_pred,squared=False))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test,y_pred))

