# coding:gbk
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from PIL import Image
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import np_utils
from keras.layers.core import Dropout, Activation
import tensorflow as tf
import random
import numpy as np


class MachineLearn():
    def __init__(self, file_path):
        BASE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        PATH = BASE_PATH + "data" + os.sep + "mnist.npz"

        self.batch_size = 128
        self.nb_classes = 10
        self.nb_epoch = 2
        self.img_size = 28 * 28

        f = np.load(PATH)
        self.x_train = f["x_train"]
        self.x_test = f["x_test"]
        self.y_train = f["y_train"]
        self.y_test = f["y_test"]
        # print(self.x_train.shape,self.y_train.shape)
        f.close()

        # 加工自变量数据，吧数据拉伸成28*28一维数组
        self.x_train = (self.x_train.reshape(self.y_train.shape[0], self.img_size).astype("float32")) / 255
        self.x_test = (self.x_test.reshape(self.y_test.shape[0], self.img_size).astype("float32")) / 255

        self.y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.nb_classes)
        # print(self.x_train.shape)
        # print('++++++++')
        # print(self.y_train.shape)

        # 加工预测数据
        # 模式L为灰色图像
        im = Image.open(file_path).resize((28, 28)).convert("L")
        im_arr = np.array(im)
        pred_img = (im_arr.reshape(self.img_size).astype("float32")) / 255
        x_pred = []
        x_pred.append(pred_img)
        self.x_pred = np.array(x_pred)

    # 构建模型
    def DNN_keras(self):
        model = Sequential([
            Dense(10, input_shape=(self.img_size,), activation="softmax")
        ])
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.nb_epoch, verbose=1,
                  validation_data=(self.x_test, self.y_test))

        # 模型预测
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        # print(score)
        metr = score[1]
        pred = model.predict(self.x_pred)
        pred = np.argmax(pred[0])
        return pred, metr

    def MLP_Keras(self):
        model = Sequential([
            Dense(512, input_shape=(self.img_size,)),
            Activation("relu"),
            Dropout(0.2),
            Dense(512, input_shape=(512,)),
            Activation("relu"),
            Dropout(0.2),
            Dense(10, input_shape=(512,), activation="softmax"),
        ])

        # 编译模型
        # 使用rmsprob改进的梯度下降算法
        # 损失函数：catgorical_crossentropy
        # 评估模型:accuracy
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.nb_epoch, verbose=1,
                  validation_data=(self.x_test, self.y_test))
        # 测量评估
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        metr = score[1]
        pred = model.predict(self.x_pred)
        pred = np.argmax(pred[0])
        return pred, metr

    def DNN_Tensorflow(self):
        # 输入占位符
        x = tf.placeholder("float", [None, 784])
        # 权重矩阵
        W = tf.Variable(tf.zeros([784, 10]))
        # 偏置矩阵
        b = tf.Variable(tf.zeros([10]))
        # 预测值
        y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
        # 真实值,输出占位符
        y = tf.placeholder("float", [None, 10])
        # 损失函数:交叉熵
        cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
        # 建立梯度下降算法
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        # 启动模型
        sess = tf.Session()
        # 激活变量
        sess.run(tf.global_variables_initializer())
        # 开始训练
        train_len = self.y_train.shape[0]
        for i in range(3000):
            ram_size1 = random.randint(1, train_len)
            ram_size2 = ram_size1 + 128
            if ram_size2 > train_len:
                ram_size2 = ram_size1
                ram_size1 = ram_size1 - 128
            x_train = self.x_train[ram_size1:ram_size2]
            y_train = self.x_train[ram_size1:ram_size2]
            sess.run(train_step, feed_dict={x: x_train, y: y_train})
        # 模型评估
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        metr = sess.run(accuracy, feed_dict={x: self.x_test, y: self.y_test})
        pred = sess.run(y_pred, feed_dict={x: self.x_pred})


        pred = np.argmax(pred[0])
        return pred, metr

# a = MachineLearn('//')
# a.DNN_keras()
