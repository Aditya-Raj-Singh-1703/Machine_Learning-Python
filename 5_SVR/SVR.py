import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y), 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Training the 5_SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(x,y)

# Predicting a new result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

# Visualising the 5_SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color='blue')
plt.title("SVR curve")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising Results in higher resolution
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color='blue')
plt.title("SVR curve")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
