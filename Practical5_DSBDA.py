import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df=pd.read_csv('Social_Network_Ads.csv')

#summary statistics
print('-------------------------------------------------------------------')
print('Describe: ',df.describe())
print('-------------------------------------------------------------------')
print('maximum: ',df.max())
print('-------------------------------------------------------------------')
print('Minimum: ',df.min())
print('-------------------------------------------------------------------')
print('Information: ',df.info())
print('-------------------------------------------------------------------')
print('Describe: ',df.describe())
print('-------------------------------------------------------------------')


#categorical to numerical conversion of data
df["Gender"].replace(["Male","Female"],["1","0"],inplace=True)


#spliting data
x = df[['Gender','Age', 'EstimatedSalary']] #independent variable / input
y = df['Purchased'] #dependent variable / output


#training model
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
model = clf.fit(X_train, Y_train)
y_test_predict = model.predict(X_test)

print('---------------------------------------y_test_predict---------------------------------------')
print(y_test_predict)


y_train_predict = model.predict(X_train)

print('---------------------------------------y_train_predict---------------------------------------')
print(y_train_predict)


print('--------------------------------------------------------------------------------------')
print("Recall score : ", recall_score(Y_test, y_test_predict))
print("accuracy score : ", accuracy_score(Y_test, y_test_predict))
print("f1 score : ", f1_score(Y_test, y_test_predict))
print("precision score :", precision_score(Y_test, y_test_predict ,zero_division=1))
print("confusion matrix :", confusion_matrix(Y_test, y_test_predict))
print('--------------------------------------------------------------------------------------')
