import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np

#loading dataset in pandas dataframe
df=pd.read_csv("student_data.csv")

#finding missing values
missing_values = df.isnull().sum()
print('------------------------------Missing values-------------------------------------')
print(missing_values)

#finding summary statistics
data_description = df.describe()
print('-----------------------------Summary statistics----------------------------------')
print(data_description)


#data_types and shape of data
data_types = df.dtypes
data_dimensions = df.shape
print('----------------------------datatypes------------------------------------------')
print(data_types)
print('-----------------------------shape--------------------------------------------')
print(data_dimensions)

#first 5 rows
head=df.head()
print('-----------------------------first 5 rows-------------------------------------')
print(head)

#last 5 rows
tail=df.tail()
print('----------------------------last 5 rows---------------------------------------')
print(tail)

#printing column name along with non null count and datatype
print('-----------------------------info---------------------------------------------')
df.info()


#categorical to numerical data conversion using replace method 
print("----------------------------------------- categorical to numerical data conversion using replace method-----------------------------------------")
copy=df["sex"].replace(["M","F"],["1","0"],inplace=False)
print('-------------------------converted dataframe-------------------------')
print(copy)
print('-----------------------Original dataframe-------------------------')
print(df)


#categorical to numerical data conversion using get Dummies method  method --> one-hot encoding
print("----------------------------------------- categorical to numerical data conversion using get_dummies method-----------------------------------------")
copy=pd.get_dummies(df["sex"])
print('---------------------------------------converted dataframe------------------------------------------------')
print(copy)
print("----------------------------------------Original dataframe---------------------------------------------------")
print(df)


#categorical to numerical data conversion using cat.codes 
print("----------------------------------------- categorical to numerical data conversion using cat.codes-----------------------------------------")
df = df.astype({"sex":"category"})
copy = df["sex"].cat.codes
print('-------------------------------------Converted dataframe------------------------------------------------ ')
print(copy)
print("----------------------------------------Original dataframe---------------------------------------------------")
print(df)


#categorical to numerical data conversion using LabelEncoder --> Label encoding
le = LabelEncoder()
df["sex_numerical"]= le.fit_transform(df["sex"])
print("----------------------------------------- categorical to numerical data conversion using label encoder fit_transform -----------------------------------------")
print("----------------------------------------Original dataframe---------------------------------------------------")
print(df)


#normalization using normalize method of preprocessing 
age=np.array(df["age"])
print("----------------------------------------- normalization using normalize method of preprocessing-----------------------------------------")
normalized_age = preprocessing.normalize([age])
print(normalized_age)



