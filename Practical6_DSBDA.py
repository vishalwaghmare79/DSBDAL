import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

#reading csv file
df = pd.read_csv("Iris.csv")

print('---------------------------------------summary statistics----------------------------------------')
print(df.describe())
print('----------------------------------shape---------------------------------')
print(df.shape)
print('----------------------------------isna---------------------------------')
print(df.isna())
print('-----------------------------------columns--------------------------------')
print(df.columns)
print('--------------------------------------isna sum-----------------------------')
print(df.isna().sum())
print('-------------------------------------------------------------------')


#splitting data
x = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = df["Species"]

#training the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
Model = GaussianNB()
Model.fit(x_train, y_train)

y_test_predict = Model.predict(x_test)

#anaylizing the model
print('---------------------------------------------------------------------------')
print("Score Of the model : ",Model.score(x_test,y_test))
print('---------------------------------------------------------------------------')
print("confusion matrix : ", confusion_matrix(y_test, y_test_predict))
print('---------------------------------------------------------------------------')
print("accuracy score : ", accuracy_score(y_test, y_test_predict))
print('---------------------------------------------------------------------------')
print("Error rate : ", 1-accuracy_score(y_test, y_test_predict))
print('---------------------------------------------------------------------------')
print("Precision score : ", precision_score(y_test, y_test_predict, average='macro'))
print('---------------------------------------------------------------------------')
print("recall score : ", recall_score(y_test, y_test_predict, average='macro'))
print('---------------------------------------------------------------------------')
