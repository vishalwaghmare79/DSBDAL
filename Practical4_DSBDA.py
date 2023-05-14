import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('housing.csv')


# Summary Statistics
print('-----------------------------------------------------summary statistics--------------------------------------------------------')
print(df.head())
print('-------------------------------------------------------------------------------')
print(df.shape)
print('-------------------------------------------------------------------------------')
print(df.info())
print('-------------------------------------------------------------------------------')
print(df.isna().sum())
print('-------------------------------------------------------------------------------')
print(df.columns)
print('-------------------------------------------------------------------------------')
print('Desciption is: ',df.describe())
print('-------------------------------------------------------------------------------')
print('minimum is: ',df.min())
print('-------------------------------------------------------------------------------')
print('maximum is: ',df.max())
print('-------------------------------------------------------------------------------')


#function to remove outliers
def Remove_outlier(df,var): 
	Q1=df[var].quantile(0.25)
	Q3=df[var].quantile(0.75)
	IQR=Q3-Q1
	df_final=df[~((df[var]<(Q1-1.5*IQR)) | (df[var]>(Q3+1.5*IQR)))]
	return df_final
	
print('------------------------------------------------------------------------------')
print('Before removing outliers : ', df.shape)
print('------------------------------------------------------------------------------')
df=Remove_outlier(df,'RM')
print('------------------------------------------------------------------------------')
print('After removing outliers : ', df.shape)
print('------------------------------------------------------------------------------')
df=Remove_outlier(df,'LSTAT')
print('------------------------------------------------------------------------------')
print('After removing outliers : ', df.shape)
print('------------------------------------------------------------------------------')

#spliting independent and dependent variables
x=df[['RM','LSTAT']] # independent variables / input
y=df['MEDV'] #dependent variable / output

#spliting  data
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=42)


#classifier
clf = LinearRegression()
model = clf.fit(X_train, Y_train) # Regression model for training data

y_test_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)

print('-----------------------------------y_test_predict--------------------------------------------')
print(y_test_predict)
print('-------------------------------------------------------------------------------')
rmse = (np.sqrt(metrics.mean_squared_error(Y_test,y_test_predict)))
r2 = metrics.r2_score(Y_test,y_test_predict)
print("The performance model for training")
print('RMSE IS : {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('-------------------------------------------------------------------------------')


print('------------------------------------y_train_predict-------------------------------------------')
print(y_train_predict)
print('-------------------------------------------------------------------------------')
rmse = (np.sqrt(metrics.mean_squared_error(Y_train,y_train_predict)))
r2 = metrics.r2_score(Y_train,y_train_predict)
print("The performance model for training")
print('RMSE IS : {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('-------------------------------------------------------------------------------')


