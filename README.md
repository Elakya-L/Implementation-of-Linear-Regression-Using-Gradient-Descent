# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Read the dataset and extract R&D Spend as X and Profit as Y.

2.Normalize X and initialize m = 0, b = 0, learning rate and epochs.

3.Use Gradient Descent to update m and b repeatedly to minimize prediction error.

4.Predict Y values using y=mx+b and plot the data points with the regression line.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Elakya L
RegisterNumber: 212225230066
*/

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
data= pd.read_csv("Startup.csv")
X=data['R&D Spend'].values 
Y=data['Profit'].values 
X=(X-X.mean())/X.std()
m,b= 0,0
learning_rate=0.01
epochs=1000
n=len(X)
for i in range(epochs):
    Y_pred= m*X + b
    dm=(-2/n)*np.sum(X*(Y-Y_pred))
    db=(-2/n)*np.sum(Y-Y_pred)
    m=m-(learning_rate*dm)
    b=b-(learning_rate*db)
print("slope(m):",m)
print("intercept(b):",b)
Y_pred= m*X+b

plt.scatter(X,Y,label="data points")
plt.plot(X,Y_pred,color="green")
plt.xlabel("R&D spend(normalised)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.legend()
plt.show()
```

## Output:

<img width="848" height="607" alt="image" src="https://github.com/user-attachments/assets/05a81cf9-9753-412e-8286-a2f7dc952537" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
