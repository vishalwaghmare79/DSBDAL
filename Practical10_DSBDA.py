import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Iris.csv")


sns.boxplot(x=df['Species'],y=df['SepalLengthCm'])
plt.show()

sns.boxplot(x=df['Species'],y=df['SepalWidthCm'])
plt.show()

sns.boxplot(x=df['Species'],y=df['PetalLengthCm'])
plt.show()

sns.boxplot(x=df['Species'],y=df['PetalWidthCm'])
plt.show()

sns.histplot(data=df, x="Species",y="SepalLengthCm")
plt.show()

sns.histplot(data=df, x="Species",y="SepalWidthCm")
plt.show()

sns.histplot(data=df, x="Species",y="PetalWidthCm")
plt.show()

sns.histplot(data=df, x="Species",y="PetalLengthCm")
plt.show()
