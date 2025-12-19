import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv("D:\Программы\PyCharm 2025.1.1.1\Projects\Preprocessing\Lab_1\starship.csv")
print(df.dtypes)
print()

x = df.drop(['CryoSleep'], axis=1)
y = df['CryoSleep']
XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size=0.4, random_state=82)

RFmodel = RandomForestClassifier(oob_score=True, max_depth=9)
RFmodel.fit(XTrain, YTrain)
YPredRF = RFmodel.predict(XTest)
RFAccuracy = accuracy_score(YTest, YPredRF)
YProbaRF = RFmodel.predict_proba(XTest)
OOBAccuracy = RFmodel.oob_score_
OOBerror = 1 - OOBAccuracy

AdaBoostModel = AdaBoostClassifier()
AdaBoostModel.fit(XTrain, YTrain)
YPredAda = AdaBoostModel.predict(XTest)
AdaAccuracy = accuracy_score(YTest, YPredAda)
YProbaAda = AdaBoostModel.predict_proba(XTest)

GradBoostModel = GradientBoostingClassifier()
GradBoostModel.fit(XTrain, YTrain)
YPredGrad = GradBoostModel.predict(XTest)
GradAccuracy = accuracy_score(YTest, YPredGrad)
YProbaGrad = GradBoostModel.predict_proba(XTest)

fprRF, tprRF, _ = roc_curve(YTest, YProbaRF[:, 1])
AucRF = auc(fprRF, tprRF)
fprAda, tprAda, _ = roc_curve(YTest, YProbaAda[:, 1])
AucAda = auc(fprAda, tprAda)
fprGrad, tprGrad, _ = roc_curve(YTest, YProbaGrad[:, 1])
AucGrad = auc(fprGrad, tprGrad)
accuracySummary = pd.DataFrame({
    'Method': ['Random Forest', 'AdaBoost', 'Gradient Boosting'],
    'Accuracy': [RFAccuracy, AdaAccuracy, GradAccuracy],
    'ROC-AUC': [AucRF, AucAda, AucGrad],
    'OOB Accuracy': [OOBAccuracy, '-', '-']
})
print(accuracySummary.to_string(index=False))

plt.figure(figsize=(10, 8))
plt.plot(fprRF, tprRF, color='blue', lw=2,
         label=f'Random Forest      (AUC = {AucRF})')
plt.plot(fprAda, tprAda, color='red', lw=2,
         label=f'AdaBoost               (AUC = {AucAda})')
plt.plot(fprGrad, tprGrad, color='green', lw=2,
         label=f'Gradient Boosting (AUC = {AucGrad})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Случайная модель')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Сравнение ROC-кривых для методов ансамблевого обучения')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()