# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Aaron I
RegisterNumber:  212223230002
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/student_scores.csv")
df

x = df.iloc[:,:-1].values
print("X-Values:",x)
y = df.iloc[:,1].values
print("Y-Values:",y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
print("X-Training Data:",x_train)
print("X-Testing Data:",x_test)
print("Y-Training Data:",y_train)
print("Y-Testing Data:",y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("Y-Predited:",y_pred)
print("Y-Testing",y_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE  = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE  = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![Screenshot 2024-08-29 113315](https://github.com/user-attachments/assets/2aeb955d-8fb0-451b-a24d-dc727c7ce1c9)

![Screenshot 2024-08-29 113327](https://github.com/user-attachments/assets/f57f2e89-8d5d-42ad-8408-b00c97637320)

![Screenshot 2024-08-29 113337](https://github.com/user-attachments/assets/72185337-b936-4edf-9735-329084d45213)

![Screenshot 2024-08-29 113348](https://github.com/user-attachments/assets/e10e2122-2f82-41ad-8f52-dc85ad3b61e4)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
