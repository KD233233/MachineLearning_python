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
        self.sex = sex  # �Ա�
        self.age = age  # ����
        self.fare = fare  # Ʊ��

        self.child = 0  # Ĭ�ϷǶ�ͯ
        if float(age) <= 10:
            self.child = 1
        if sex == "female":
            self.Sex_female = 1
            self.Sex_male = 0
        else:
            self.Sex_female = 0
            self.Sex_male = 1

        # �������ݼ�
        PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep
        # print(PATH)
        file_PATH = PATH + "train.csv"
        data_train = pd.read_csv(file_PATH)
        # print(data_train.Sex.describe())
        # ����Ageȱʧֵ����ƽ��ֵ
        age_mean = data_train.Age.dropna().mean()
        data_train.loc[data_train.Age.isnull(), "Age"] = age_mean
        df = data_train  # ��������ǰ
        # print("����ǰ",df.head())
        scaler = preprocessing.StandardScaler() #ȥ��ֵ�ͷ����һ����

        Age_temp = data_train["Age"].values.reshape(-1, 1) # df.column.values   ��array��ʽ����ָ��column������ȡֵ
        # print(data_train["Age"].values)
        Fare_temp = data_train["Fare"].values.reshape(-1, 1)

        # ��һ��
        df["Age_scaled"] = scaler.fit_transform(Age_temp)
        # print("�����",df.head())
        df["Fare_scaled"] = scaler.fit_transform(Fare_temp)

        self.age = (int(self.age) - scaler.mean_) / scaler.var_ ** 0.5
        self.fare = (int(self.fare) - scaler.mean_) / scaler.var_ ** 0.5

        # �Ա��������·���
        dummies_sex = pd.get_dummies(data_train["Sex"], prefix="Sex")
        # print(dummies_sex)

        # �������仮��
        df["Child"] = (data_train.Age <= 10).astype(int) #����С��10����1�����򷵻�0

        # �ϲ�������������
        df = pd.concat([df, dummies_sex], axis=1)
        # print("�ϲ��������",df.head())
        # ��ȡ�µ�ѵ������
        self.train_df = df.filter(regex="Survived|Age_.*|Fare_.*|Sex_.*|Child")
        # print("�µ����ݼ�",self.train_df.head())
        # �µĲ�������
        self.pred = [[self.age, self.fare, self.child, self.Sex_female, self.Sex_male]]

        # �ָ�����
        Y = self.train_df.iloc[:, 0]
        X = self.train_df.iloc[:, 1:]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

    def Ai(self,Model):
        regr = Model
        regr.fit(self.X_train, self.Y_train)
        Y_pred = regr.predict(self.X_test)
        metr = metrics.accuracy_score(self.Y_test, Y_pred)  # ģ����ȷ��
        pred = regr.predict(self.pred)  # Ԥ��չʾҪ�õĽ��
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
    #     metr = metrics.accuracy_score(self.Y_test, Y_pred)  # ģ����ȷ��
    #     pred = regr.predict(self.pred)  # Ԥ��չʾҪ�õĽ��
    #
    #
    #
    #
    # def LogicRegression(self):
    #     regr = LogisticRegression
    #     regr.fit(self.X_train,self.Y_train)
    #     Y_pred = regr.predict(self.X_test)
    #     metr = metrics.accuracy_score(self.Y_test, Y_pred)  # ģ����ȷ��
    #     pred = regr.predict(self.pred)  # Ԥ��չʾҪ�õĽ��
    #     # print(pred)
    #     # print(metr)
    #     return pred, metr
    # def Decis


#
# m = MachineLearn("female",20,200)
# m.LoginRegression()
