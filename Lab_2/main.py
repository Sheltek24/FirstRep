import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("D:\Программы\PyCharm 2025.1.1.1\Projects\Preprocessing\Lab_1\starship.csv")
print(df.dtypes)
print()

x = df.drop(['Age'], axis=1)
y = df['Age']
XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size=0.4, random_state=12)

polyReg = PolynomialFeatures(5)
model = ElasticNet(alpha=0.1, l1_ratio=0.01)
XTrain = polyReg.fit_transform(XTrain)
XTest = polyReg.fit_transform(XTest)
model.fit(XTrain, YTrain)

YPredTest = model.predict(XTest)
RMSE = root_mean_squared_error(YTest, YPredTest)
print("RMSE полиномиальной регрессии: ", RMSE)
print("Теоретическое стандартное отклонение: ", y.std())


x = df.drop(['CryoSleep'], axis=1)
y = df['CryoSleep']
XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size=0.4, random_state=82)

logModel = LogisticRegression(max_iter=1000)
logModel.fit(XTrain, YTrain)
YPredTest = logModel.predict(XTest)
confMatrix = confusion_matrix(YTest, YPredTest)
print(confMatrix)
print()
report = classification_report(YTest, YPredTest)
print(report)



