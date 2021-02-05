import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training RFR Model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

print(regressor.predict([[6.5]]))

# Visualising -> Same as Decision Tree Regression