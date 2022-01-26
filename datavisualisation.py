import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset = pd.read_csv("/Users/bhuvanrj/Desktop/git upload/UberDS.csv")
X = dataset.dispatching_base_number
Y = dataset.active_vehicles
Z = dataset.trips
D = dataset.date



plt.style.use("ggplot")
plt.subplot(211)
plt.bar(X,Y,label="BnvAv")
plt.title("Active Vehicles per Base")
plt.xlabel("Dispatch Base no.")
plt.ylabel("No. of Active vehicles")

plt.subplot(212)
plt.bar(D,Z,label="DvT",width=0.3)
plt.title("No. of Trips per Day")
plt.xlabel("Date")
plt.ylabel("No. of trips")
plt.show()
