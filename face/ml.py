# coding:gbk

from sklearn.linear_model import LinearRegression  # 线性
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.cluster as sc
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import scale, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import  Dense

# from keras.models import Sequential
# from keras.layers import Dense

class MachineLearn(object):
    def __init__(self, file_path, k):

        BASE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        PATH = BASE_PATH + 'att_faces' + os.sep
        X = []
        Y = []
        # DNN
        XNN = []
        YNN = []

        # 将照片导入到numpy数组，然后将他们的像素矩阵转换为向量
        for dir_path, dir_names, file_names in os.walk(PATH):
            for fn in file_names:
                if fn[-3:] == 'pgm':
                    image_filename = os.path.join(dir_path, fn)
                    im = Image.open(image_filename).convert('L')
                    im_arr = np.array(im)
                    # print(im_arr)
                    x_im_arr = scale(im_arr.reshape(10304).astype('float32'))
                    X.append(x_im_arr)
                    Y.append(dir_path)

                    XNN.append(im_arr.reshape(10304).astype('float32') / 255)
                    YNN.append(dir_path)
        X = np.array(X)
        # 分离出训练集和测试集
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y)
        # 降维处理
        pca = PCA(n_components=k)  # 降维
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)

        # 带预测的样本
        im = Image.open(file_path).convert('L')
        im_arr = np.array(im)
        test_face = scale(im_arr.reshape(10304).astype('float32'))

        x_pred = []
        x_pred.append(test_face)
        x_pred = np.array(x_pred)
        self.x_pred = pca.transform(x_pred)

        # DNN :one-hot 独热编码
        values = np.array(YNN)
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(values)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        self.YNN_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        # print(self.YNN_onehot_encoded)
        inverted = self.label_encoder.inverse_transform([np.argmax(self.YNN_onehot_encoded[0, :])])
        # print(inverted)

        # DNN
        # 划分训练和测试数据
        test_size = int(len(self.YNN_onehot_encoded) * 0.8)
        self.XNN_train = XNN[0:test_size]
        self.XNN_test = XNN[test_size:]
        self.YNN_train = self.YNN_onehot_encoded[0:test_size]
        self.YNN_test = self.YNN_onehot_encoded[test_size:]

        # 待预测数据
        im = Image.open(file_path).convert('L')
        im_arr = np.array(im)
        test_face_NN = scale(im_arr.reshape(10304).astype('float32'))

        x_pred_NN = []
        x_pred_NN.append(test_face_NN)
        # x_pred_NN = np.array(x_pred)
        self.x_pred_NN = pca.transform(x_pred)

    def Ai(self, Model):
        regr = Model
        regr.fit(self.X_train, self.Y_train)
        Y_pred = regr.predict(self.X_test)
        metr = metrics.accuracy_score(self.Y_test, Y_pred)
        pred = regr.predict(self.x_pred)
        return pred, metr

    def KD(self):
        KNN = self.Ai(Model=KNeighborsClassifier(n_neighbors=5))
        LogicRegression = self.Ai(Model=LogisticRegression(C=1000))
        DecisionTree = self.Ai(Model=DecisionTreeClassifier(max_depth=40))
        RandomForest = self.Ai(Model=RandomForestClassifier(max_depth=4, n_estimators=400, random_state=7))
        SVM = self.Ai(Model=SVC(kernel='linear', gamma=0.01, probability=True,class_weight='balanced'))
        # KMeans = self.Ai(Model=sc.KMeans(n_clusters=4), pred=a)

        return KNN, LogicRegression, DecisionTree, RandomForest, SVM

    def DNN_keras(self):
        model = Sequential(
            [Dense(10,input_shape=(10304,),activation="softmax"),]
        )

        model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
        nb_epoch = 10

        batch_size = 128
        model.fit(self.XNN_train,self.YNN_train,batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_data=(self.XNN_test,self.YNN_test))

        score = model.evaluate(self.XNN_test,self.YNN_test,verbose=0)

        metr = score[1]
        pred = model.predict(self.x_pred_NN)

        pred = self.label_encoder.inverse_transform([np.argmax(pred[0])])

        return pred,metr

#
# c = MachineLearn('F:\\python人工智能\\AI-day (17)\\AI-day (17)\\test\\10.pgm',5)
# c.DNN_keras()