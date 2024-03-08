# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

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
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: YUVARAJ B
RegisterNumber:  21222040186
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
![simple linear regression model for predicting the marks scored](sam.png)
![Screenshot (127)](https://github.com/yuvarajmonarch/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/a01d3729-77d6-42fd-9183-08ad68eb058c)
![Screenshot (128)](https://github.com/yuvarajmonarch/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/a16999d7-991b-4ff1-b52b-d2f3bf67d7da)
![Screenshot (129)](https://github.com/yuvarajmonarch/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/ef37f0fc-12a8-4140-b4bc-e4e676120754)
![Screenshot (130)](https://github.com/yuvarajmonarch/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/d8fe899e-8e4d-4831-9a87-e1bee6bb2b22)
![Screenshot (131)](https://github.com/yuvarajmonarch/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/cd648c4c-afbd-499e-a3ac-dd15b6a1f538)
![Screenshot (132)](https://github.com/yuvarajmonarch/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122221735/d01421dd-a1de-4590-9be5-0e96a3262207)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
