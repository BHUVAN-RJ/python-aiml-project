import pandas as pd
import numpy as ny
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

df=pd.DataFrame(data.data, columns = data.feature_names)
df['target']=data.target


X=df.iloc[:,:-1].values
Y=df.iloc[:,30].values



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

from sklearn.preprocessing import MinMaxScaler
scalar =MinMaxScaler()

scalar.fit(X_train)
scalar.fit(X_test)

X_train_scaled=scalar.transform(X_train)
X_test_scaled=scalar.transform(X_test)



from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled,Y_train)

//predicting using our model
y_pred =model.predict(X_test_scaled)

X_test_unscaled=scalar.inverse_transform(X_test_scaled)
X_train_unscaled=scalar.inverse_transform(X_train_scaled)


//Metrics to compare effeciency of our regressor model
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(Y_test,y_pred))
print("model score",model.score(X_test_scaled,Y_test))









