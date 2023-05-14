import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Read the dataset into a DataFrame
df = pd.read_csv("weather.csv")

# Print the DataFrame
print(df)

# Print the column names
print(df.columns)

# Print the first few rows of the DataFrame
print(df.head())

# Print the shape of the DataFrame
print(df.shape)

# Get information about the DataFrame
print(df.info())

# Get descriptive statistics of the DataFrame
print(df.describe())

# Check for missing values in the DataFrame
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Check for missing values again
print(df.isnull().sum())

# Get unique values of the 'RainTomorrow' column
df['RainTomorrow'].unique()

# Extract the target variable
Y = df.RainTomorrow
print(Y.head())

from sklearn import preprocessing

# Perform label encoding on the 'RainTomorrow' column
label_encoder = preprocessing.LabelEncoder()
df['RainTomorrow'] = label_encoder.fit_transform(df['RainTomorrow'])
print(df['RainTomorrow'].unique())

# Perform label encoding on the 'WindGustDir' column
label_encoder = preprocessing.LabelEncoder()
df['WindGustDir'] = label_encoder.fit_transform(df['WindGustDir'])
print(df['WindGustDir'].unique())

# Perform label encoding on the 'RainToday' column
label_encoder = preprocessing.LabelEncoder()
df['RainToday'] = label_encoder.fit_transform(df['RainToday'])
print(df['RainToday'].unique())

# Create a heatmap of the correlation matrix
hm = sn.heatmap(data=df.corr(), annot=True, annot_kws={'size': 8})
sn.set(rc={'figure.figsize': (12, 12)})
plt.show()

# Create a histogram of the 'MaxTemp' column
sn.displot(df['MaxTemp'])
plt.show()

# Create a box plot of the 'MinTemp' column
sn.boxplot(df['MinTemp'])
plt.show()

# Create a box plot of the 'MaxTemp' column
sn.boxplot(df['MaxTemp'])
plt.show()

# Create a count plot of the 'MaxTemp' column
sn.countplot(data=df, x='MaxTemp')
plt.show()

# Create the feature matrix X by dropping irrelevant columns
X = df.drop(['RainTomorrow', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am'], axis='columns')
print(X.head())

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

from sklearn import svm

# Create a linear SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

from sklearn import metrics

# Calculate the accuracy of the classifier
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))

# Create a polynomial SVM classifier
model = svm.SVC(kernel='poly')
model.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the classifier
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))

