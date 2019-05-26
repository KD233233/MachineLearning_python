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


class MachineLearn(object):
    def __init__(self, sex, age, fare):
        self.sex = sex  # 性别
        self.age = age  # 年龄
        self.fare = fare  # 票价

        self.child = 0  # 默认非儿童
        if float(age) <= 10:
            self.child = 1
        if sex == "female":
            self.Sex_female = 1
            self.Sex_male = 0
        else:
            self.Sex_female = 0
            self.Sex_male = 1

        # 导入数据集
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        # print(PATH)
        file_PATH = PATH + "train.csv"
        data_train = pd.read_csv(file_PATH)
        # print(data_train.Sex.describe())
        # 处理Age缺失值，赋平均值
        age_mean = data_train.Age.dropna().mean()
        data_train.loc[data_train.Age.isnull(), "Age"] = age_mean
        df = data_train  # 不做处理前
        # print("处理前",df.head())
        scaler = preprocessing.StandardScaler() #去均值和方差归一化化

        Age_temp = data_train["Age"].values.reshape(-1, 1) # df.column.values   以array形式返回指定column的所有取值
        # print(data_train["Age"].values)
        Fare_temp = data_train["Fare"].values.reshape(-1, 1)

        # 归一化
        df["Age_scaled"] = scaler.fit_transform(Age_temp)
        # print("处理后",df.head())
        df["Fare_scaled"] = scaler.fit_transform(Fare_temp)

        self.age = (int(self.age) - scaler.mean_) / scaler.var_ ** 0.5
        self.fare = (int(self.fare) - scaler.mean_) / scaler.var_ ** 0.5

        # 性别特征重新分类
        dummies_sex = pd.get_dummies(data_train["Sex"], prefix="Sex")
        # print(dummies_sex)

        # 年龄区间划分
        df["Child"] = (data_train.Age <= 10).astype(int) #年龄小于10返回1，否则返回0

        # 合并新整理数据列
        df = pd.concat([df, dummies_sex], axis=1)
        # print("合并后的数据",df.head())
        # 获取新的训练数据
        self.train_df = df.filter(regex="Survived|Age_.*|Fare_.*|Sex_.*|Child")
        # print("新的数据集",self.train_df.head())
        # 新的测试数据
        self.pred = [[self.age, self.fare, self.child, self.Sex_female, self.Sex_male]]

        # 分割数据
        Y = self.train_df.iloc[:, 0]
        X = self.train_df.iloc[:, 1:]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

    def Ai(self,Model):
        regr = Model
        regr.fit(self.X_train, self.Y_train)
        Y_pred = regr.predict(self.X_test)
        metr = metrics.accuracy_score(self.Y_test, Y_pred)  # 模型正确率
        pred = regr.predict(self.pred)  # 预测展示要用的结果
        return pred, metr
    def KD(self):
        KNN = self.Ai(Model=KNeighborsClassifier(n_neighbors=20),)
        LogicRegression = self.Ai(Model=LogisticRegression(C=1000))
        DecisionTree = self.Ai(Model=DecisionTreeClassifier(max_depth=4))
        RandomForest = self.Ai(Model=RandomForestClassifier(max_depth=4,n_estimators=400,random_state=7))
        SVM = self.Ai(Model=SVC(kernel='rbf',gamma=0.01,probability=True))
        return KNN,LogicRegression,DecisionTree,RandomForest,SVM

    # def KNN(self):
    #     regr = KNeighborsClassifier(n_neighbors=20)
    #     regr.fit(self.X_train, self.Y_train)
    #     Y_pred = regr.predict(self.X_test)
    #     metr = metrics.accuracy_score(self.Y_test, Y_pred)  # 模型正确率
    #     pred = regr.predict(self.pred)  # 预测展示要用的结果
    #
    #
    #
    #
    # def LogicRegression(self):
    #     regr = LogisticRegression
    #     regr.fit(self.X_train,self.Y_train)
    #     Y_pred = regr.predict(self.X_test)
    #     metr = metrics.accuracy_score(self.Y_test, Y_pred)  # 模型正确率
    #     pred = regr.predict(self.pred)  # 预测展示要用的结果
    #     # print(pred)
    #     # print(metr)
    #     return pred, metr
    # def Decis


#
# m = MachineLearn("female",20,200)
# m.LoginRegression()
