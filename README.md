# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Anbu Vinotha.S
RegisterNumber: 212223230015

import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
  


## Output:

### data.head()

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118680361/b2470a2b-b0df-432f-9025-9e98aa62463d)

### data.info()

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118680361/18d380b7-c9f4-4ab2-b96d-e5d8a4aa1194)

### isnull() and sum()

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118680361/4b508f80-bc55-4780-af7d-047c18b2b46e)

### data.head() for salary

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118680361/45c42bb4-7316-42f1-ae18-a70f5192f60e)

### MSE value

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118680361/ab3376dd-0945-4b7c-94bf-82b6118ffebe)

### r2 value

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118680361/56399d9e-f0cd-4945-8a87-e5474e8cd45a)

### data prediction

![image](https://github.com/Safeeq-Fazil/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118680361/e0396bff-f3f7-4dda-a562-6b748703919b)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
