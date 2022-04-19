import numpy as np
from sklearn import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def find_beta(x_train,y_train):
    x = np.array([1]*len(x_train))
    x = np.insert(x_train , 0, values = 1, axis = 1)
    # x = np.insert(x_train , 0, values = x, axis = 1)
    beta = np.linalg.inv(x.T @ x) @ x.T @ y_train
    return beta

with open("iris_x.txt") as f:
    list = []
    for i in range(150):
        list.append(np.array(f.readline().split(), dtype=float))
    data = np.array(list)

# 開啟資料標籤檔案
with open("iris_y.txt") as f:
    label = np.array(f.read().split(), dtype=int)
# 切割資料，指定 random_state = 20220413
x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=20220413)
# 使用線性迴歸類別

# 擬合
clt = LinearRegression()
clt.fit(x_train, y_train)
print(clt.intercept_)
print(clt.coef_)


beta = find_beta(x_train,y_train)
print(beta)  