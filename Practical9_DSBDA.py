import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

df = sns.load_dataset('titanic')

print('------------------------head-------------------------')
print(df.head())
print('-----------------------shape--------------------------')
print(df.shape)
print('------------------------describe-------------------------')
print(df.describe())
print('------------------------info-------------------------')
print(df.info())
print('-------------------------------------------------')
print("--------------------isNA----------------")
print(df.isna())
print("--------------------isNA---------------")
print(df.size)
print('---------------------dtypes----------------------------')
print(df.dtypes)
print('--------------------------tail-----------------------')
print(df.tail())
print('-------------------------isnull------------------------')
print(df.isnull().sum())
print('-----------------------columns--------------------------')
print(df.columns)
print('-------------------------------------------------')


sns.boxplot(y=df['fare'] )
plt.show()

sns.boxplot( x = df['sex'], y = df['age'])
plt.show()

sns.boxplot( x = df['sex'], y = df['age'], hue = df['survived'])
plt.show()
