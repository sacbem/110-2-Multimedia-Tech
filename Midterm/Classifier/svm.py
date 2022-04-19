# -*- coding: utf-8 -*-

import cv2
import os
import struct
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def get_mnist_train(path= "./",flatten=True,kind='train'):

    # label_file = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    # img_file = os.path.join(path,'%s-images.idx3-ubyte'% kind)

    with open('train-labels.idx1-ubyte', 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file,dtype=np.uint8)

    with open('train-images.idx3-ubyte', 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)  # uint8
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, labels

def get_mnist_test(path="./",flatten=True, kind='t10k'):

    label_file = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    img_file = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)  # int8
 
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)  # uint8
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, labels  

train_image,train_labels=get_mnist_train()


vidcap = cv2.VideoCapture("test_dataset.avi")
success = True

# 存放幀的 list
video_frame = []

while success:
    # 讀入幀
    success, image = vidcap.read()

    # 確認還有沒有幀
    if not success:
        break

    # 將影像由 RGB 轉成 GRAY（這樣只會保留明亮度，所以陣列只有二維，RGB 會出現三維不好處理）
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 加到 list 裡面
    video_frame.append(image)

data = np.array(video_frame).reshape((len(video_frame), -1))

# 標準化
x_train,x_test,y_train,y_test = train_test_split(data,train_labels,train_size=0.7)
#X=preprocessing.StandardScaler().fit_transform(train_image)  # 標準化

# 模型訓練
model_svc = svm.SVC()
model_svc.fit(x_train,y_train)

#預測結果
#x=preprocessing.StandardScaler().fit_transform(test_images) # 標準化

predicted = model_svc.predict(x_test)
# 評分
#y_pred = model_svc.predict(x_test)
print(model_svc.score(x_test,y_test))
print(classification_report(y_test,predicted))