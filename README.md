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
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yashwanth Raja Durai
RegisterNumber:  21222040184
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/LENOVO/Downloads/student_scores.csv")
df.head()

df.tail()X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)


```

## Output:
![Screenshot (67)](https://github.com/yashwanthrajadurai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462316/e751fe7f-ec30-48e0-a99b-5737aee164c8)
![Screenshot (68)](https://github.com/yashwanthrajadurai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462316/89d3d404-817e-49d7-87d2-4aa024ea596c)
![Screenshot (69)](https://github.com/yashwanthrajadurai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462316/6beba6dd-f14e-4a88-a715-acb67520e95c)
![Screenshot (70)](https://github.com/yashwanthrajadurai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462316/7c13d56e-cf42-400b-a495-1259ff6f5782)
![Screenshot (71)](https://github.com/yashwanthrajadurai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462316/1ac474e6-b9dd-4020-83a4-4ee94d3c1518)
![Screenshot (72)](https://github.com/yashwanthrajadurai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462316/5d5ae391-975b-4894-9e85-e75c2b7e29fd)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
