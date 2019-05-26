# coding:gbk

from sklearn.linear_model import LinearRegression  # 线性
import numpy as np
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


# from sklearn.


class MachineLearn():

    def __init__(self):
        # 导入数据集
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        # print(PATH)
        file_PATH = PATH + "iris.data"
        self.__df = pd.read_csv(file_PATH, names=['calyx_h', 'calyx_w', 'petal_h', 'petal_w', 'class'])
        # X = self.__df.loc[:, ['petal_w']]
        # Y = self.__df.loc[:, ['petal_h']]  # 用宽度预测长度

        # self.X = self.__df.filter(regex='petal_w|petal_h|calyx_w|calyx_h')
        self.X = self.__df.loc[:, ['calyx_h', 'calyx_w', 'petal_h', 'petal_w']]
        Y = self.__df.loc[:, ['class']]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, Y, test_size=0.2,random_state=50)


        # self.x = [[5.1, 3.5, 1.4, 0.2],
        #             [4.9, 3., 1.4, 0.2],
        #             [4.7, 3.2, 1.3, 0.2],
        #             [4.6, 3.1, 1.5, 0.2],
        #             [5., 3.6, 1.4, 0.2],
        #             [5.4, 3.9, 1.7, 0.4],
        #             [4.6, 3.4, 1.4, 0.3],
        #             [5., 3.4, 1.5, 0.2],
        #             [4.4, 2.9, 1.4, 0.2],
        #             [4.9, 3.1, 1.5, 0.1],
        #             [5.4, 3.7, 1.5, 0.2],
        #             [4.8, 3.4, 1.6, 0.2],
        #             [4.8, 3., 1.4, 0.1],
        #             [4.3, 3., 1.1, 0.1],
        #             [5.8, 4., 1.2, 0.2],
        #             [5.7, 4.4, 1.5, 0.4],
        #             [5.4, 3.9, 1.3, 0.4],
        #             [5.1, 3.5, 1.4, 0.3],
        #             [5.7, 3.8, 1.7, 0.3],
        #             [5.1, 3.8, 1.5, 0.3],
        #             [5.4, 3.4, 1.7, 0.2],
        #             [5.1, 3.7, 1.5, 0.4],
        #             [4.6, 3.6, 1., 0.2],
        #             [5.1, 3.3, 1.7, 0.5],
        #             [4.8, 3.4, 1.9, 0.2],
        #             [5., 3., 1.6, 0.2],
        #             [5., 3.4, 1.6, 0.4],
        #             [5.2, 3.5, 1.5, 0.2],
        #             [5.2, 3.4, 1.4, 0.2],
        #             [4.7, 3.2, 1.6, 0.2],
        #             [4.8, 3.1, 1.6, 0.2],
        #             [5.4, 3.4, 1.5, 0.4],
        #             [5.2, 4.1, 1.5, 0.1],
        #             [5.5, 4.2, 1.4, 0.2],
        #             [4.9, 3.1, 1.5, 0.1],
        #             [5., 3.2, 1.2, 0.2],
        #             [5.5, 3.5, 1.3, 0.2],
        #             [4.9, 3.1, 1.5, 0.1],
        #             [4.4, 3., 1.3, 0.2],
        #             [5.1, 3.4, 1.5, 0.2],
        #             [5., 3.5, 1.3, 0.3],
        #             [4.5, 2.3, 1.3, 0.3],
        #             [4.4, 3.2, 1.3, 0.2],
        #             [5., 3.5, 1.6, 0.6],
        #             [5.1, 3.8, 1.9, 0.4],
        #             [4.8, 3., 1.4, 0.3],
        #             [5.1, 3.8, 1.6, 0.2],
        #             [4.6, 3.2, 1.4, 0.2],
        #             [5.3, 3.7, 1.5, 0.2],
        #             [5., 3.3, 1.4, 0.2],
        #             [7., 3.2, 4.7, 1.4],
        #             [6.4, 3.2, 4.5, 1.5],
        #             [6.9, 3.1, 4.9, 1.5],
        #             [5.5, 2.3, 4., 1.3],
        #             [6.5, 2.8, 4.6, 1.5],
        #             [5.7, 2.8, 4.5, 1.3],
        #             [6.3, 3.3, 4.7, 1.6],
        #             [4.9, 2.4, 3.3, 1.],
        #             [6.6, 2.9, 4.6, 1.3],
        #             [5.2, 2.7, 3.9, 1.4],
        #             [5., 2., 3.5, 1.],
        #             [5.9, 3., 4.2, 1.5],
        #             [6., 2.2, 4., 1.],
        #             [6.1, 2.9, 4.7, 1.4],
        #             [5.6, 2.9, 3.6, 1.3],
        #             [6.7, 3.1, 4.4, 1.4],
        #             [5.6, 3., 4.5, 1.5],
        #             [5.8, 2.7, 4.1, 1.],
        #             [6.2, 2.2, 4.5, 1.5],
        #             [5.6, 2.5, 3.9, 1.1],
        #             [5.9, 3.2, 4.8, 1.8],
        #             [6.1, 2.8, 4., 1.3],
        #             [6.3, 2.5, 4.9, 1.5],
        #             [6.1, 2.8, 4.7, 1.2],
        #             [6.4, 2.9, 4.3, 1.3],
        #             [6.6, 3., 4.4, 1.4],
        #             [6.8, 2.8, 4.8, 1.4],
        #             [6.7, 3., 5., 1.7],
        #             [6., 2.9, 4.5, 1.5],
        #             [5.7, 2.6, 3.5, 1.],
        #             [5.5, 2.4, 3.8, 1.1],
        #             [5.5, 2.4, 3.7, 1.],
        #             [5.8, 2.7, 3.9, 1.2],
        #             [6., 2.7, 5.1, 1.6],
        #             [5.4, 3., 4.5, 1.5],
        #             [6., 3.4, 4.5, 1.6],
        #             [6.7, 3.1, 4.7, 1.5],
        #             [6.3, 2.3, 4.4, 1.3],
        #             [5.6, 3., 4.1, 1.3],
        #             [5.5, 2.5, 4., 1.3],
        #             [5.5, 2.6, 4.4, 1.2],
        #             [6.1, 3., 4.6, 1.4],
        #             [5.8, 2.6, 4., 1.2],
        #             [5., 2.3, 3.3, 1.],
        #             [5.6, 2.7, 4.2, 1.3],
        #             [5.7, 3., 4.2, 1.2],
        #             [5.7, 2.9, 4.2, 1.3],
        #             [6.2, 2.9, 4.3, 1.3],
        #             [5.1, 2.5, 3., 1.1],
        #             [5.7, 2.8, 4.1, 1.3],
        #             [6.3, 3.3, 6., 2.5],
        #             [5.8, 2.7, 5.1, 1.9],
        #             [7.1, 3., 5.9, 2.1],
        #             [6.3, 2.9, 5.6, 1.8],
        #             [6.5, 3., 5.8, 2.2],
        #             [7.6, 3., 6.6, 2.1],
        #             [4.9, 2.5, 4.5, 1.7],
        #             [7.3, 2.9, 6.3, 1.8],
        #             [6.7, 2.5, 5.8, 1.8],
        #             [7.2, 3.6, 6.1, 2.5],
        #             [6.5, 3.2, 5.1, 2.],
        #             [6.4, 2.7, 5.3, 1.9],
        #             [6.8, 3., 5.5, 2.1],
        #             [5.7, 2.5, 5., 2.],
        #             [5.8, 2.8, 5.1, 2.4],
        #             [6.4, 3.2, 5.3, 2.3],
        #             [6.5, 3., 5.5, 1.8],
        #             [7.7, 3.8, 6.7, 2.2],
        #             [7.7, 2.6, 6.9, 2.3],
        #             [6., 2.2, 5., 1.5],
        #             [6.9, 3.2, 5.7, 2.3],
        #             [5.6, 2.8, 4.9, 2.],
        #             [7.7, 2.8, 6.7, 2.],
        #             [6.3, 2.7, 4.9, 1.8],
        #             [6.7, 3.3, 5.7, 2.1],
        #             [7.2, 3.2, 6., 1.8],
        #             [6.2, 2.8, 4.8, 1.8],
        #             [6.1, 3., 4.9, 1.8],
        #             [6.4, 2.8, 5.6, 2.1],
        #             [7.2, 3., 5.8, 1.6],
        #             [7.4, 2.8, 6.1, 1.9],
        #             [7.9, 3.8, 6.4, 2.],
        #             [6.4, 2.8, 5.6, 2.2],
        #             [6.3, 2.8, 5.1, 1.5],
        #             [6.1, 2.6, 5.6, 1.4],
        #             [7.7, 3., 6.1, 2.3],
        #             [6.3, 3.4, 5.6, 2.4],
        #             [6.4, 3.1, 5.5, 1.8],
        #             [6., 3., 4.8, 1.8],
        #             [6.9, 3.1, 5.4, 2.1],
        #             [6.7, 3.1, 5.6, 2.4],
        #             [6.9, 3.1, 5.1, 2.3],
        #             [5.8, 2.7, 5.1, 1.9],
        #             [6.8, 3.2, 5.9, 2.3],
        #             [6.7, 3.3, 5.7, 2.5],
        #             [6.7, 3., 5.2, 2.3],
        #             [6.3, 2.5, 5., 1.9],
        #             [6.5, 3., 5.2, 2.],
        #             [6.2, 3.4, 5.4, 2.3],
        #             [5.9, 3., 5.1, 1.8]]
        # self.y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #             0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        #             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        #             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        #
        # self.petal_w = petal_w  # 花瓣宽度
        # self.petal_h = petal_h  # 花瓣长度
        # self.calyx_w = calyx_w  # 花萼宽度
        # self.calyx_h = calyx_h
        # x = np.array(self.x).reshape(-1,1)
        # y = np.array(self.y)
        #
        # self.x0 = np.array(self.x)[:,0].reshape(-1,1)
        # # print(self.petal_w)
        # self.y0 = np.array(self.y)
        # # print(y)
        #
        # self.petal_w = np.array(self.petal_w).reshape(-1,1)
        #
        #
        # # self.petal_w,self.petal_h,self.calyx_w,self.calyx_h=x[:0],x[:1],x[:2],x[:3]
        # # self.petal_w = np.array(self.x[:,0])
        #
        # # self.petal_w = np.reshape(1, -1)
        # self.pred = [[self.petal_w,self.petal_h,self.calyx_w,self.calyx_h]]
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    def Linear(self, petal_w):
        X = self.__df.loc[:, ['petal_w']]
        Y = self.__df.loc[:, ['petal_h']]  # 用宽度预测长度
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        # print(X_train)
        # print(Y_train)
        model = LinearRegression()
        model.fit(X_train, Y_train)
        pred = model.predict(np.array(petal_w).reshape(-1, 1))  # 应用测试
        # print(pred)
        return pred

    # a = MachineLearn()
    # a.Linear(np.array(2).reshape(-1,1))

    def Ai(self, Model, pred=0):
        X = self.__df.filter(regex='petal_w|petal_h|calyx_w|calyx_h')
        # print(X)
        Y = self.__df.loc[:, ['class']]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=50)
        regr = Model
        regr.fit(X_train, Y_train)
        Y_pred = regr.predict(X_test)
        metr = metrics.accuracy_score(Y_test, Y_pred)  # 模型正确率
        pred = regr.predict(pred)  # 预测展示要用的结果
        # print(pred + '-*----------------')
        # print(metr)
        return pred, metr

    def KD(self, a):
        KNN = self.Ai(Model=KNeighborsClassifier(n_neighbors=5), pred=a)
        LogicRegression = self.Ai(Model=LogisticRegression(C=1000), pred=a)
        DecisionTree = self.Ai(Model=DecisionTreeClassifier(max_depth=4), pred=a)
        RandomForest = self.Ai(Model=RandomForestClassifier(max_depth=4, n_estimators=400, random_state=7), pred=a)
        SVM = self.Ai(Model=SVC(kernel='rbf', gamma=0.01, probability=True), pred=a)
        # KMeans = self.Ai(Model=sc.KMeans(n_clusters=4), pred=a)

        return KNN, LogicRegression, DecisionTree, RandomForest, SVM

    def KMeans(self,pred=[[1,1,1,1]]):
        # X = self.__df.filter(regex='petal_w|petal_h|calyx_w|calyx_h')
        model = sc.KMeans(n_clusters=3)
        model.fit(self.X)
        pred = model.predict(pred)
        #轮廓系数
        metr = metrics.silhouette_score(self.X,model.labels_)

        if pred ==  [1]:
            pred = np.array('Iris-virginica',dtype=object)
        elif pred == [0]:
            pred = np.array('Iris-setosa', dtype=object)
        else:
            pred = np.array('Iris-versicolor', dtype=object)
        # centens = model.cluster_centers_

        # metr = metrics.accuracy_score(self.Y_test, Y_pred)  # 模型正确率
        # print(pred,metr)
        return [pred] ,metr


a =MachineLearn()
b = a.KMeans(pred=[[6,2,3,2.2]])
print(str(b[0][0]))
print(type(b[1]))
print(type(b[0]))
# b = a.Ai(Model=LogisticRegression(C=1000),pred=[[1.2,2.3,4.5,2]])
# print(b)
# print(b[0])
# print(b[0][0])
