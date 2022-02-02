import pandas as pd
import numpy as np

dataset = pd.read_csv('/Users/bhuvanrj/Desktop/git upload/UberDS.csv')
cols =['dispatching_base_number','date','active_vehicles']
dataset["date"] = [float(str(i).replace("/", "")) for i in dataset["date"]]
X= dataset[cols]
Y=dataset.trips
#data preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A = make_column_transformer((OneHotEncoder(categories="auto"),[0]),remainder='passthrough')
X =A.fit_transform(X)
#data training
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=1)
#data scaling
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()

scale.fit(X_train)

X_train_scaled=scale.transform(X_train)
X_test_scaled=scale.transform(X_test)

#Logistic regression model
from sklearn.neighbors import KNeighborsClassifier
kmodel=KNeighborsClassifier(n_neighbors=5)
kmodel.fit(X_train,Y_train)

y_pred=kmodel.predict(X_test_scaled)

#confusion matrix
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(Y_test,y_pred)
print(cnf_matrix)

#Heatmap for confusion matrix
from matplotlib import pyplot as plt
import seaborn as sns

sns.heatmap(pd.DataFrame(cnf_matrix))
plt.title("Confusion matrix heat map")
plt.xlabel("Actual data")
plt.ylabel("Predicted data")
plt.show()

#Accuracy score
from sklearn import metrics
errors = metrics.mean_squared_error(Y_test, y_pred, squared=False)
print(errors)
