# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: JEGAN S K
RegisterNumber:  212225230117
*/

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
X=data.data[:,:3]
Y=np.column_stack((data.target,data.data[:,6]))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
SS=StandardScaler()
X_train=SS.fit_transform(X_train)
Y_train=SS.fit_transform(Y_train)
X_test=SS.fit_transform(X_test)
Y_test=SS.fit_transform(Y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_reg=MultiOutputRegressor(sgd)

multi_output_reg.fit(X_train,Y_train)
Y_pred=multi_output_reg.predict(X_test)
Y_pred=SS.inverse_transform(Y_pred)
Y_test=SS.inverse_transform(Y_test)

print("Predictions : \n",Y_pred[:8])
```

## Output:

<img width="1262" height="221" alt="image" src="https://github.com/user-attachments/assets/47235be1-853c-4644-aa58-a5c34e0624ae" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
