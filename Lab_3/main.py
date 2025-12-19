import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, roc_curve, auc
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("D:\Программы\PyCharm 2025.1.1.1\Projects\Preprocessing\Lab_1\starship.csv")
print(df.dtypes)
print()

x = df.drop(['Age'], axis=1)
y = df['Age']
XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size=0.4, random_state=12)

regTreeModel = DecisionTreeRegressor(max_depth=11)
regTreeModel.fit(XTrain, YTrain)
YPred = regTreeModel.predict(XTest)

RMSE = root_mean_squared_error(YTest, YPred)
print("RMSE дерева решений для регрессии: ", RMSE)
print("Теоретическое стандартное отклонение: ", y.std())
plt.figure(figsize=(20, 15))
plot_tree(regTreeModel, filled=True, feature_names=x.columns, rounded=True, fontsize=10)
plt.title("Дерево решений для регрессии")


x = df.drop(['CryoSleep'], axis=1)
y = df['CryoSleep']
XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size=0.4, random_state=82)

classTreeModel = DecisionTreeClassifier(max_depth=8)
classTreeModel.fit(XTrain, YTrain)
YProba = classTreeModel.predict_proba(XTest)
fpr, tpr, thresholds = roc_curve(YTest, YProba[:, 1])
rocAuc = auc(fpr, tpr)
print("Площадь под ROC-кривой: ", rocAuc)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {rocAuc})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайная модель')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-кривая для классификации')
plt.legend(loc="lower right")
plt.grid(True)

plt.figure(figsize=(20, 15))
plot_tree(classTreeModel, filled=True, feature_names=x.columns, class_names=['No CryoSleep', 'CryoSleep'], rounded=True, fontsize=10)
plt.title("Дерево решений для классификации")

plt.show()