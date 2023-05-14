import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

#for Employee_Salary_Dataset
df = pd.read_csv('Employee_Salary_Dataset.csv')

#printing sumarry statistics information for numeric attributes

print('-----------------------------------------statistics for numeric columns----------------------------------------------')
for col in df.columns:
	if pd.api.types.is_numeric_dtype(df[col]):
		print('----------------------------sumarry statistics for {}------------------------------------------'.format(col))
		print('Minimum value is: ',df[col].min())
		print('Maximum value is: ' ,df[col].max())
		print('Mean value is: ' ,df[col].mean())
		print('Median value is: ' ,df[col].median())
		print('Mode value is: ' ,df[col].mode())
		print('Standard deviation is: ', df[col].std())
		print('Variance is: ' ,df[col].var())
		print('Skewness is: ' ,df[col].skew())
		print('Kurtosis is: ' ,df[col].kurt())
		print('Desciption is: ' ,df[col].describe())


# Group by
print('----------------------------------------------------------------------------------------------------------------------------')
print("min of salary grouped by gender: ",df['Salary'].groupby(df['Gender']).min())
print('----------------------------------------------------------------------------------------------------------------------------')
print("max of salary grouped by gender: ",df['Salary'].groupby(df['Gender']).max())
print('----------------------------------------------------------------------------------------------------------------------------')
print("mean of salary grouped by gender: ",df['Salary'].groupby(df['Gender']).mean())
print('----------------------------------------------------------------------------------------------------------------------------')
print("median of salary grouped by gender: ",df['Salary'].groupby(df['Gender']).median())
print('----------------------------------------------------------------------------------------------------------------------------')
print("std of salary grouped by gender: ",df['Salary'].groupby(df['Gender']).std())
print('----------------------------------------------------------------------------------------------------------------------------')
print("var of salary grouped by gender: ",df['Salary'].groupby(df['Gender']).var())
print('----------------------------------------------------------------------------------------------------------------------------')
print("skewness of salary grouped by gender: ",df['Salary'].groupby(df['Gender']).skew())
print('----------------------------------------------------------------------------------------------------------------------------')
print("desciption of ID grouped by gender: ",df['ID'].groupby(df['Gender']).describe())
print('----------------------------------------------------------------------------------------------------------------------------')
print("desciption of Experience_Years grouped by gender: ",df['Experience_Years'].groupby(df['Gender']).describe())
print('----------------------------------------------------------------------------------------------------------------------------')
print("desciption of age grouped by gender: ",df['Age'].groupby(df['Gender']).describe())
print('----------------------------------------------------------------------------------------------------------------------------')
print("desciption of salary grouped by gender: ",df['Salary'].groupby(df['Gender']).describe())
print('----------------------------------------------------------------------------------------------------------------------------')

plt.boxplot(df['Salary']) #boxplot or whisker's plot
plt.show()

sns.scatterplot(data = df, x = 'Salary', y = 'Age') #scatterplot
plt.show()

print('----------------------------------------------------------------------------------------------------------------------------')
print("percentile is : ")
print(np.percentile(df['Salary'],75))




#for Iris dataset
df = pd.read_csv('Iris.csv')

print('-----------------------------------statistics--------------------------------------------')
for col in df.columns:
	print('---------------------------statistics for {}--------------------------------------'.format(col))
	if pd.api.types.is_numeric_dtype(df[col]):
		print('Minimum value is: ',df[col].min())
		print('Maximum value is: ' ,df[col].max())
		print('Mean value is: ' ,df[col].mean())
		print('Median value is: ' ,df[col].median())
		print('Standard deviation is: ', df[col].std())
		print('Variance is: ' ,df[col].var())
		print('Skewness is: ' ,df[col].skew())
		print('Kurtosis is: ' ,df[col].kurt())
		print('Desciption is: ' ,df[col].describe())
		
		
print('-------------------------------group by species for each col-------------------------------------')
for col in df.columns:
	print('-----------------------------group by statistics for {}--------------------------------'.format(col))
	if pd.api.types.is_numeric_dtype(df[col]):
		print("min of ",col," grouped by Species: ",df[col].groupby(df['Species']).min())
		print("max of ",col," grouped by Species: ",df[col].groupby(df['Species']).max())
		print("mean of ",col," grouped by Species: ",df[col].groupby(df['Species']).mean())
		print("median of ",col," grouped by Species: ",df[col].groupby(df['Species']).median())
		print("std of ",col," grouped by Species: ",df[col].groupby(df['Species']).std())
		print("var of ",col," grouped by Species: ",df[col].groupby(df['Species']).var())
		print("skewness of ",col," grouped by Species: ",df[col].groupby(df['Species']).skew())


print('-----------------------------------quantile--------------------------------------')
for col in df.columns:
	print('--------------------------quantiles for {}------------------------------'.format(col))
	if pd.api.types.is_numeric_dtype(df[col]):
		print("Quartile of ",col," 25% : ",df[col].quantile(0.25))
		print("Quartile of ",col," 50% : ",df[col].quantile(0.50))
		print("Quartile of ",col," 75% : ",df[col].quantile(0.75))
		
		
		
print('------------------------------------box plot for each column-------------------------------------')
for col in df.columns:
	if pd.api.types.is_numeric_dtype(df[col]):
		plt.boxplot(df[col])
		plt.show()
		
		
sns.scatterplot(data = df, x = 'SepalLengthCm', y = 'SepalWidthCm')
plt.show()
