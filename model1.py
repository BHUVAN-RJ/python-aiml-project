import pandas as pd
import numpy as np

dataset = pd.read_csv('/Users/bhuvanrj/Desktop/git upload/UberDS.csv')
cols =['dispatching_base_number','date','active_vehicles']
dataset["date"] = [float(str(i).replace("/", "")) for i in dataset["date"]]
X= dataset[cols]
Y=dataset.trips

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A = make_column_transformer((OneHotEncoder(categories="auto"),[0]),remainder='passthrough')
X =A.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=1)

from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)



