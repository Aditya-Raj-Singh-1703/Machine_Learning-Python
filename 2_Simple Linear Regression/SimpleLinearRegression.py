# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# No missing data in dataset hence no taking care
# No categorical data hence no encoding

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature Scaling is not required in this model.

# Training the Simple Linear Regression model on Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test results
y_predict = regressor.predict(x_test)

# Visualising the Training set Results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train))
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test Set Results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train))
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# making a single prediction
single_test = regressor.predict([[12]])
print(single_test)

# getting final regression equation with the values of coefficients and intercept
coefficient = regressor.coef_
intercept = regressor.intercept_

# Equation will be:
# Salary = coefficient + years of experience*intercept

