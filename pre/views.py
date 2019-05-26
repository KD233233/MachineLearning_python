#coding:gbk
from django.shortcuts import render
from django.http import JsonResponse
from .quant import Quant
import pandas as pd


# Create your views here.

def index(request):
    return render(request, "index6.html")


def setcode(request):
    code = request.POST.get("code", None)
    quant = Quant()
    stock_list = quant.get_realtime_stock_info(code)
    msg = "<table width=490px><tr><td>代码</td><td>名称</td><td>开盘价</td><td>上一天收盘价</td><td>价格</td><td>获取数据</td></tr>"
    for i in range(len(stock_list)):
        row_dict = stock_list[i]
        button_str = "<input type=button value='提取' id=begin" + str(i) + " onclick='quantAction(" + str(i) + ")' >"
        msg += "<tr><td>" + row_dict["code"] + "</td><td>" + row_dict["name"] + "</td><td>" + row_dict[
            "open"] + "</td><td>" + row_dict["pre_close"] + "</td><td>" + row_dict[
                   "price"] + "</td><td>" + button_str + "</td></tr>"
    msg += "</table>"
    return JsonResponse({"msg": msg})


def catch(request):
    start_time = request.POST.get("start_time", None)
    end_time = request.POST.get("end_time", None)
    catch_code = request.POST.get("catch_code", None)
    quant = Quant()
    quant.catch_data(catch_code, start_time, end_time)
    return JsonResponse({"msg": "获取" + catch_code + "成功"})


def outcsv(request):
    catch_code = request.POST.get("catch_code", None)
    quant = Quant()
    stock_path = quant.outcsv(catch_code)
    return JsonResponse({"msg": stock_path})


def dodata(request):
    catch_code = request.POST.get("catch_code", None)
    quant = Quant()
    a = quant.dodata(catch_code)
    tran_len = len(a[0])
    test_len = len(a[1])
    return JsonResponse({"msg": "训练数据：" + str(tran_len) + " 测试数据：" + str(test_len)})


def pred(request):
    catch_code = request.POST.get("catch_code", None)
    logic_select = request.POST.get("logic_select", None)
    quant = Quant()
    a = quant.dodata(catch_code)

    res_pred = ""

    if logic_select == "LinearRegression":
        res_pred = quant.LinearRegression()
    elif logic_select == "SVR":
        res_pred = quant.SVR()
    return JsonResponse({"msg": str(res_pred)})


def back(request):
    catch_code = request.POST.get("catch_code", None)
    quant_select = request.POST.get("quant_select", None)
    quant = Quant()
    a = quant.dodata(catch_code)
    pre = quant.SVR()
    res_pred = ""

    if quant_select == "daily":
        res_pred = quant.daily_stats()
    elif quant_select == "id_":
        res_pred = quant.id_stats()
    elif quant_select == "overnight":
        res_pred = quant.overnight_stats()
    elif quant_select == "custom_":
        res_pred = quant.custom_stats()
    return JsonResponse({"msg": str(res_pred)})