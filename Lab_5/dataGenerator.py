import numpy as np

np.random.seed(47)

X = np.random.randint(0, 2, size=(100, 12))
Y = np.array([[0, 1] if row.sum() <= 4 else [1, 0]  for row in X]) # сумма ответов меньше 4 победа оппозиции, больше партии

np.savetxt("D:\Программы\PyCharm 2025.1.1.1\Projects\Preprocessing\Lab_5\dataIn.txt", X, fmt="%d")
np.savetxt("D:\Программы\PyCharm 2025.1.1.1\Projects\Preprocessing\Lab_5\dataOut.txt", Y, fmt="%d")