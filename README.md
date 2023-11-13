# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import the standard Libraries.
 
 2.Set variables for assigning dataset values. 
 
 3.Import linear regression from sklearn.
 
 4.Assign the points for representing in the graph. 
 
 5.Predict the regression for marks by using the representation of the graph. 
 
 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: MADHUMITHA M

RegisterNumber: 212222220020

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('student_scores.csv')

print(df)

df.head(0)

df.tail(0)

print(df.head())

print(df.tail())

x = df.iloc[:,:-1].values

print(x)

y = df.iloc[:,1].values

print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

print(y_test)

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')

plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.title("Hours vs Scores(Training set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='black')

plt.plot(x_train,regressor.predict(x_train),color='red')

plt.title("Hours vs Scores(Testing set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

mse=mean_absolute_error(y_test,y_pred)

print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)

print('MAE = ',mae)

rmse=np.sqrt(mse)

print("RMSE= ",rmse)


## Output:

df.head():

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/6aeec917-c56c-4741-9741-26f42dbfa9e7)

df.tail():

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/c127470f-fab0-414d-be01-67df80d4ff09)

Array value of X:

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/c70beb04-7682-426b-bc05-05a4ff5f863c)

Array value of y:

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/939108fe-4bc5-4647-ba57-b17c81d9a902)

Values of Y prediction:

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/3ecd163a-8a80-463a-bbdb-bb93848c474c)

Array values of Y test:

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/0c045513-a925-49b6-b339-5e9ec953d0a2)

Training test graph:

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/c49a589e-2d68-4e2a-bf8e-572430197aac)

Test set graph:

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/90a30d69-3adb-4bf2-a800-6847a73729da)

Values of MSE,MAE,RMSE:

![image](https://github.com/Madhumithamahendran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394403/b5f5b6ae-9cda-44a2-b252-fb8d51e4ea89)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
