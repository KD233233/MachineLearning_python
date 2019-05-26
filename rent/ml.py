from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV,KFold
import sklearn.preprocessing as preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

class Machinelearn(object):
    def __init__(self,dist,area,beds,rooms,ori):
        self.dist = dist
        self.area = area
        self.beds = beds
        self.rooms = rooms
        self.ori = ori

        #导入数据
        PATH = os.path.abspath(os.path.dirname(__file__))+os.sep+"data"+os.sep+"mydata.csv"
        print(PATH)
        data_train = pd.read_csv(PATH,names=["cs_dist","cs_size","cs_type","cs_rent","cs_ori","beds","rooms"])
        data_new = data_train[["cs_rent","cs_dist","cs_size","cs_ori","beds","rooms"]]
        #归一化
        scaler = preprocessing.StandardScaler()
        size_temp = data_new["cs_size"].values.reshape(-1,1)
        data_new["size_temp"] = scaler.fit_transform(size_temp)
        self.area = (self.area-scaler.mean_)/scaler.var_**0.5
        #区域特征重新分类
        dummies_dist = pd.get_dummies(data_new["cs_dist"],prefix="Dist")
        #方向特征重新分类
        dummies_ori = pd.get_dummies(data_new["cs_ori"],prefix="Ori")
        df = pd.concat([data_new,dummies_dist,dummies_ori],axis=1)
        print(df.head())
        #用来训练的数据
        self.trainf_df = df.filter(regex="cs_rent|beds|rooms|size_temp|Dist_.*|Ori_.*")
        print(self.trainf_df.head())
        #用来预测的数据
        self.Ori_east = 0
        self.Ori_south = 0
        self.Ori_west = 0
        self.Ori_north = 0
        self.Dist_haizhu =0
        self.Dist_liwan =0
        self.Dist_panyu =0
        self.Dist_tianhe =0
        self.Dist_yuexiu =0
        if self.ori == "east":
            self.Ori_east = 1
            self.Ori_north = 0
            self.Ori_west = 0
            self.Ori_south =0
        elif self.ori == "south":
            self.Ori_south = 1
            self.Ori_west = 0
            self.Ori_north = 0
            self.Ori_east = 0
        elif self.ori == "north":
            self.Ori_north = 1
            self.Ori_east = 0
            self.Ori_west = 0
            self.Ori_south = 0
        elif self.ori == "west":
            self.Ori_west = 1
            self.Ori_south = 0
            self.Ori_east = 0
            self.Ori_north = 0

        if self.Dist_haizhu == "tianhe":
            self.Dist_haizhu = 0
            self.Dist_liwan = 0
            self.Dist_panyu = 0
            self.Dist_tianhe = 1
            self.Dist_yuexiu = 0
        elif self.Dist_haizhu == "yuexiu":
            self.Dist_haizhu = 0
            self.Dist_liwan = 0
            self.Dist_panyu = 0
            self.Dist_tianhe = 0
            self.Dist_yuexiu = 1
        elif self.Dist_haizhu == "liwan":
            self.Dist_haizhu = 0
            self.Dist_liwan = 1
            self.Dist_panyu = 0
            self.Dist_tianhe = 0
            self.Dist_yuexiu = 0
        elif self.Dist_haizhu == "haizhu":
            self.Dist_haizhu = 1
            self.Dist_liwan = 0
            self.Dist_panyu = 0
            self.Dist_tianhe = 0
            self.Dist_yuexiu = 0
        elif self.Dist_haizhu == "panyu":
            self.Dist_haizhu = 0
            self.Dist_liwan = 0
            self.Dist_panyu = 1
            self.Dist_tianhe = 0
            self.Dist_yuexiu = 0

        #预测数据生成
        self.pred = [[self.beds,self.rooms,self.area,self.Dist_haizhu,self.Dist_liwan,self.Dist_panyu,self.Dist_tianhe,self.Dist_yuexiu,self.Ori_east,self.Ori_north,self.Ori_south,self.Ori_west]]

        #训练数据的分割
        y = self.trainf_df.iloc[:,0]
        X = self.trainf_df.iloc[:,1:]
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(X,y,test_size=0.2,random_state=7)



    def LinearRegression(self):
        #建模
        regr = LinearRegression()
        #训练
        regr.fit(self.X_train,self.Y_train)
        #预测
        Y_pred = regr.predict(self.X_test)
        metr = metrics.r2_score(self.Y_test,Y_pred)
        pred = regr.predict(self.pred)
        return pred,metr

    def SVR(self):
        regr = SVR(kernel="linear")
        regr.fit(self.X_train,self.Y_train)
        Y_pred = regr.predict(self.X_test)
        metr = metrics.r2_score(self.Y_test,Y_pred)
        pred = regr.predict(self.pred)
        return pred,metr

    def DecisionTree(self):
        regr = DecisionTreeRegressor()
        regr.fit(self.X_train, self.Y_train)
        Y_pred = regr.predict(self.X_test)
        metr = metrics.r2_score(self.Y_test, Y_pred)
        pred = regr.predict(self.pred)
        return pred, metr

    def KNN(self):
        regr = KNeighborsRegressor(n_neighbors=5)
        regr.fit(self.X_train, self.Y_train)
        Y_pred = regr.predict(self.X_test)
        metr = metrics.r2_score(self.Y_test, Y_pred)
        pred = regr.predict(self.pred)
        return pred, metr

    def RandomForest(self):
        regr = RandomForestRegressor(n_estimators=50)
        regr.fit(self.X_train, self.Y_train)
        Y_pred = regr.predict(self.X_test)
        metr = metrics.r2_score(self.Y_test, Y_pred)
        pred = regr.predict(self.pred)
        return pred, metr

    def LoginRegression(self):
        regr = LogisticRegression()
        regr.fit(self.X_train, self.Y_train)
        Y_pred = regr.predict(self.X_test)
        metr = metrics.r2_score(self.Y_test, Y_pred)
        pred = regr.predict(self.pred)
        return pred, metr



# dist,area,beds,rooms,ori

# m = Machinelearn("yuexiu",110.5,2,2,"男")
# l = m.DecisionTree()
# print(l)
