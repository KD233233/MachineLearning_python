# coding:gbk

import tushare as ts
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
import os
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


class Quant(object):
    def __init__(self):
        self.PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep + "data" + os.sep

    def get_realtime_stock_info(self, code):
        codes = code.split(" ")
        df = ts.get_realtime_quotes(codes)
        df = df[["code", "name", "open", "price", "high", "low", "volume", "amount", "pre_close"]]
        results = []
        for i in range(len(df)):
            row_temp = df.iloc[i, :]
            col_name = df.ix[i, ["name"]].to_dict()
            col_code = df.ix[i, ["code"]].to_dict()
            col_open = df.ix[i, ["open"]].to_dict()
            col_price = df.ix[i, ["price"]].to_dict()
            col_pre_close = df.ix[i, ["pre_close"]].to_dict()
            stock_real_info_dict = dict(col_name, **col_code, **col_open, **col_price, **col_pre_close)
            results.append(stock_real_info_dict)
        return results

    def catch_data(self, code, start_time, end_time):
        df = ts.get_k_data(code, start=start_time, end=end_time)
        engine = create_engine("mysql://root:233233@127.0.0.1:3306/quant")
        df.to_sql("stock" + code, engine, if_exists="replace", index=False)
        return "stock" + code

    def out_csv(self, code):
        conn = pymysql.connect("localhost", port=3306, user="root", password=233233, db="quant")
        sql = "select data,open,close from stock" + code
        result = pd.read_sql(sql, conn)
        stock_path = self.PATH + "stock" + code + ".csv"
        result.to_csv(stock_path, encoding='utf-8')
        return stock_path

    def dadata(self, code):
        stock_path = self.PATH + "stock" + code + ".csv"
        self.spy = pd.read_csv(stock_path)

        # 预测当天收盘价
        # 从当前日开20建模，开始预测
        for i in range(1, 21, 1):
            self.spy.loc[:, "Close Minus" + str(i)] = self.spy["close"].shift(i)

        self.sp20 = self.spy.filter(regex="^close$|^(Close Minus).*")
        # 把列次序反向排列
        self.sp20 = self.sp20.iloc[:, ::-1]
        # 删除前20行数据
        self.sp20 = self.sp20.iloc[20:, ]

        # 训练和测试
        stock_count = len(self.sp20)
        # 80%用于测试
        train_size = int(stock_count * 0.8)
        self.X_train = self.sp20[:train_size]
        self.y_train = self.sp20["close"].shift(-1)[:train_size]

        self.X_test = self.sp20[train_size:]
        self.y_test = self.sp20["close"].shift(-1)[train_size:]

        aaaa = np.c_[self.X_train, self.y_train]
        bbbb = pd.DataFrame(aaaa)
        return self.X_train, self.X_test

    def SVR(self):
        regr = SVR(kernel="linear")
        regr.fit(self.X_train, self.y_train)
        self.preds = regr.predict(self.X_test)
        return self.preds[len(self.preds) - 1]

    def LinearRession(self):
        regr = LinearRegression()
        regr.fit(self.X_train, self.y_train)
        self.preds = regr.predict(self.X_test)
        return self.preds[len(self.preds) - 1]

    # 统计信息方法
    # s是某一个策略的收益率
    def get_stat(self, s, n=252):
        # 删除NaN数据
        s = s.dropna()
        # 盈利次数
        wins = len(s[s > 0])
        # 亏损次数
        losses = len(s[s < 0])
        # 盈亏平衡次数，收益率等于0
        evens = len(s[s == 0])
        # 盈利平均值：收益率大于零的平均值
        mean_w = round(s[s > 0].mean(), 3)
        # 亏损平均值：收益率小于零的平均值
        mean_l = round(s[s < 0].mean(), 3)
        # 盈利和亏损的比例
        win_r = round(wins / losses, 3)
        # 平均收益，收益率的平均值
        mean_trd = round(s.mean(), 3)
        # 收益率标准差
        sd = round(np.std(s), 3)
        # 最大亏损：收益率最小
        max_l = s.min()
        # 最大盈利
        max_w = s.max()
        # 夏普比率：收益率平均值/（收益率方差/n)**0.5
        sharpe_r = (s.mean() / np.std(s) * (n ** 0.5))
        # 交易次数
        cnt = len(s)
        stats_res = (
            "交易次数：", cnt,
            "盈利次数：", wins,
            "亏损次数：", losses,
            "盈亏平衡次数：", evens,
            "盈利平均值：", mean_w,
            "亏损平均值：", mean_l,
            "盈利与亏损的比例：", win_r,
            "平均收益：", mean_trd,
            "标准差：", win_r,
            "最大亏损：", max_l,
            "最大盈利：", max_w,
            "夏普比率：", sharpe_r,
        )
        return stats_res

    def daily_stats(self):
        #日内交易
        #日内交易收益率
        self.spy["Daily Change Rate"] = pd.Series(self.spy["close"]-self.spy["open"])/self.spy["open"]
        return self.get_stat(self.spy["Daily Change Rate"])

    def id_stats(self):
        #日间交易
        #日间交易收益率
        self.spy["id Cjange Rate"] = pd.Series((self.spy["close"]-self.spy["close"].shift(1)))/self.spy["close"].shift(1)*100
        return self.get_stat(self.spy["id Cjange Rate"])

    def overnight_stats(self):
        #隔夜交易
        #隔夜交易收益率
        self.spy["Overnight Change Rate"] = pd.Series((self.spy["open"]-self.spy["close"].shift(1)))/self.spy["close"].shift(1)*100
        return self.get_stat(self.spy["Overnight Change Rate"])

    #自定义新的量化交易策略
    def get_signal(self,r):
        if(r["open"])<r["Predicted Next Close"]:
            return 1
        else:
            return  0

    #收益率
    def get_ret(self,r):
        if r["Signal"]==1:
            return ((r["Next Day Close"]-r["open"])/r["open"])*100
        else:
            return 0




    def custom_stats(self):
        tf = pd.DataFrame(list(zip(self.y_test,self.preds)),columns=["Next Day Close","Predicted Next Close"],index=self.y_test.index)
        #当前日收益价
        cdc = self.sp20[["close"]].iloc[1000:]
        #下一日的开盘价
        ndo = self.spy[["open"]].iloc[1020:].shift(-1)

        ccc = pd.merge(tf,cdc,left_index=True,right_index=True)
        ddd = pd.merge(ccc,ndo,left_index=True,right_index=True)
        tf2 = ddd.assign(Signal=ddd.apply(self.get_signal,axis=1))
        tf3 = tf2.assign(PnL=tf2.apply(self.get_ret,axis=1))
        return self.get_stat(tf3["PnL"])




# q = Quant()
# code = "000001 600000"
# print(q.get_realtime_stock_info(code))

# q = Quant()
# print(q.catch_data("000001","2019-04-01","2019-05-05"))









