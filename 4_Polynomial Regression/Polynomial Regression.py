import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression Model on the whole dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

# Training the Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the Linear Regression Results
plt.scatter(x, y, color='red')
plt.plot(x,regressor.predict(x), color='blue')
plt.title('Linear Regression (Truth or Bluff)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Visualising the Polynomial Regression Results
plt.scatter(x, y, color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title("Polynomial Regression")
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Predicting a new result with Linear Regression
print(regressor.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
