import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#this code is to check for missing data
dataset = pd.read_csv("/Users/bhuvanrj/Desktop/git upload/UberDS.csv")
sns.heatmap(dataset.isnull())
plt.show()
