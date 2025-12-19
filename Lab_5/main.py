import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

x = np.loadtxt(r"D:\Программы\PyCharm 2025.1.1.1\Projects\Preprocessing\Lab_5\dataIn.txt")
y = np.loadtxt(r"D:\Программы\PyCharm 2025.1.1.1\Projects\Preprocessing\Lab_5\dataOut.txt")
XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size=0.4, random_state=14)

scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform(XTest)

model = keras.Sequential([
    keras.layers.Dense(12, activation='sigmoid', input_shape=(12,)),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(XTrain, YTrain, epochs=100, batch_size=12, validation_data=(XTest, YTest))

YPred = np.argmax(model.predict(XTest), axis=1)
YTrue = np.argmax(YTest, axis=1)
accuracy = accuracy_score(YTrue, YPred)
print(f"Neural Network Accuracy: {accuracy}")

YTrainLabel = np.argmax(YTrain, axis=1)
YTestLabel = np.argmax(YTest, axis=1)
RFModel = RandomForestClassifier(n_estimators=9)
RFModel.fit(XTrain, YTrainLabel)
YPredRF = RFModel.predict(XTest)
RFAccuracy = accuracy_score(YTestLabel, YPredRF)
print(f"Random Forest accuracy: {RFAccuracy}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()