# Project1_Online_News_Popularity
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('OnlineNewsPopularity.csv')
#df.drop('url',axis=1)
df.shape

corr = df.corr()
sns.heatmap(corr)

x = df.iloc[:,1:-1].values
y = df.shares
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print('Accuracy is: ',accuracy_score(y_pred,y_test))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Scaling the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)
#sc.fit_transform(y)

# Applying Logistic Regression

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print('Accuracy is: ',accuracy_score(y_pred,y_test))

plt.scatter(y_pred,y_test)

# Applying Linear Regression

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn import metrics
lr1 = LinearRegression()

1y_pred = lr.predict(X_test)
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df1 = pd.DataFrame({'Expected:':y_test,'Predicted:':y_pred})
df1

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test,check_input=True)

#Accuracy
print(clf.score(X_test,y_test, sample_weight=None))
print(metrics.accuracy_score(y_pred,y_test))


print(cross_val_score(clf, X_train, y_train, cv=10))
