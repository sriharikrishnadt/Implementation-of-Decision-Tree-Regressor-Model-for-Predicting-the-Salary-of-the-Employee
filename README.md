# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SRI HARI KRISHNA D T
RegisterNumber: 212224240160
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
*/
```

## Output:
## Initial dataset:

![image](https://github.com/user-attachments/assets/d31e029d-f2fe-4633-864d-decdf53d2ed5)

## Data Info:

![image](https://github.com/user-attachments/assets/704650b0-69e6-4f40-a70e-db3c8292b074)

## Optimization of null values:

![image](https://github.com/user-attachments/assets/ce8441e6-fb64-47ae-a705-cbad79e38704)

## Converting string literals to numerical values using label encoder:

![image](https://github.com/user-attachments/assets/3eb94f42-a2c1-439a-aae0-0575f961fa7a)

## Assigning x and y values:

![image](https://github.com/user-attachments/assets/558edc79-2351-448c-9fa4-0281ed132296)

## Mean Squared Error:

![image](https://github.com/user-attachments/assets/150ef377-e74f-4cd5-89af-13a541206469)

## R2 (variance):

![image](https://github.com/user-attachments/assets/53fdc5da-9843-4a92-830d-c2d7dbe5e764)

## Prediction:

![image](https://github.com/user-attachments/assets/9f729dbb-9fad-4b2c-b28c-b5932ff51071)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
