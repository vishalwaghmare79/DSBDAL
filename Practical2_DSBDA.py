import pandas as pd
import scipy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Student_performance.csv')


print('-----------------------------------------------counting null values-----------------------------------------------')
print(df.isna().sum())

#filling null values with 0
print('--------------------------------------------filling null values with mean----------------------------------------')
df['VisITedResources'] = df['VisITedResources'].fillna(df['VisITedResources'].mean())
print(df['VisITedResources'])

#drawing boxplot on discussion attribute
sns.boxplot(df['Discussion'])
plt.show()


#finding correlation
print('--------------------------------finding correlation for heatmap---------------------------------------------')
print("correlation is : ",df.select_dtypes('number').corr())

#drawing heatmap --> annot=true for displaying corr value on heatmap
sns.heatmap(df.select_dtypes('number').corr(),annot=True)
plt.show()

#printing name of each column
print('---------------------------------columns-----------------------------------------------')
print(df.columns)

#boxplot on discussion , nationality and gender variable
sns.boxplot(data = df, x = 'Discussion', y = 'NationalITy', hue = 'gender')
plt.show()

#scatterplot on raisehands and visitedResources variable
sns.scatterplot(data = df, x = 'raisedhands', y = 'VisITedResources')
plt.show()

#removing outliers 
print('--------------------------------dataframe befor removing outliers----------------------------')
print(df)

#calculating lowerlimit and upperlimit for removing outliers
lower_limit=df['VisITedResources'].mean()-3*df['VisITedResources'].std()
upper_limit=df['VisITedResources'].mean()+3*df['VisITedResources'].std()

#selecting only those values that are in limit 
df1 = df[(df['VisITedResources']>lower_limit)&(df['VisITedResources']<upper_limit
)]

print('----------------------------------dataframe after removing outliers----------------------------')
print("Result : ", df1)



# Z-Score Method
def outliers_z_score(xs):
	threshold = mean_x = np.mean(xs)
	std_x = np.std(xs)
	z_scores=[(x-mean_x)/std_x for x in xs]
	return np.where(np.abs(z_scores)>threshold)
	
#Z = (x - μ) / σ

#Where:

    #Z is the Z-score
    #x is the data point
    #μ is the mean of the dataset
    #σ is the standard deviation of the dataset
	
print('------------------------------------printing outliers---------------------------------------')
output = outliers_z_score(df['Discussion'])
print(output)

sns.displot(data=df,x='VisITedResources',kind='kde')
plt.show()   
