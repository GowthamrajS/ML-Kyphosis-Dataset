
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea
import missingno

dataset = pd.read_csv("kyphosis.csv")

missingno.matrix(dataset,figsize = (6,5))


dataset.info()

dataset.describe()

x = dataset.drop(["Kyphosis"],axis =1)

y = dataset["Kyphosis"]


plt.figure(figsize = (10,15))
plt.subplot(2,1,1)
sea.countplot(data=x,x = "Start")

plt.subplot(2,1,2)
x["Age"].hist()


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

y = le.fit_transform(y)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y, test_size = 0.2)



from sklearn.tree import DecisionTreeClassifier as df

df = df()

df.fit(x_train,y_train)

df.score(x_train,y_train)


imp = pd.DataFrame(df.feature_importances_, index = x_train.columns,
                   columns = ["Important"]).sort_values("Important",ascending=False)


y_pred = df.predict(x_test)



from sklearn.metrics import classification_report, confusion_matrix
cr =  classification_report (y_test,y_pred)
cm =    confusion_matrix(y_test,y_pred)

sea.heatmap(cm,annot =True,fmt="g",cbar = False)



from sklearn.ensemble import RandomForestClassifier as rf

rf =rf(n_estimators=100)

rf.fit(x_train,y_train)


y_rf = rf.predict(x_test)


#Calculating X_train data how much Accuracy fitted in the model

from sklearn.metrics import classification_report, confusion_matrix
cr =  classification_report (y_test,y_rf)
cm =    confusion_matrix(y_test,y_rf)

sea.heatmap(cm,annot =True,fmt="g",cbar = False)

