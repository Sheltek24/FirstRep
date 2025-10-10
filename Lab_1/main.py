import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("titanic.csv")
print(df)
print()

nanMatrix = df.isnull()
print('До заполнения')
print(nanMatrix.sum())
print()
columnsMode = ['Cabin', 'HomePlanet', 'Cabin', 'Destination', 'Name']
for col in columnsMode:
    df[col] = df[col].fillna(df[col].mode()[0])
columnsMedian = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in columnsMedian:
    df[col] = df[col].fillna(df[col].median())
columnsMean = ['CryoSleep', 'VIP']
for col in columnsMean:
    df[col] = df[col].fillna(df[col].mean())
nanMatrix = df.isnull()
print('После заполнения')
print(nanMatrix.sum())
print()

scaler = MinMaxScaler()
columnsNorm = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
print(df['RoomService'])
print('Максимальное значение: ', max(df['RoomService']))
print()
for col in columnsNorm:
    df[col] = scaler.fit_transform(df[[col]])
print(df['RoomService'])
print('Максимальное значение: ', max(df['RoomService']))
print()

df = pd.get_dummies(df, columns=['Cabin', 'HomePlanet', 'Cabin', 'Destination', 'Name'], drop_first=True)
nanMatrix = df.isnull()
print('После OHE-преобразования')
print(nanMatrix.sum())
print()