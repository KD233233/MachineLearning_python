from django.shortcuts import render
from django.http import JsonResponse
from .ml import Machinelearn

# Create your views here.

def index(request):
    return render(request,"rent/index.html")

def pred(request):
    area = float(request.POST.get("area",None))
    beds = int(request.POST.get("beds",None))
    rooms = int(request.POST.get("rooms",None))
    ori = request.POST.get("ori",None)
    logic_select = request.POST.get("logic_select",None)
    dist = request.POST.get("dist",None)

    ml = Machinelearn(dist,area,beds,rooms,ori)

    res_pred = ""
    if logic_select == "LinearRegression":
        res_pred = ml.LinearRegression()
    elif logic_select == "SVR":
        res_pred = ml.SVR()
    elif logic_select == "DecisionTree":
        res_pred = ml.DecisionTree()
    elif logic_select == "KNN":
        res_pred = ml.KNN()
    elif logic_select == "RandomForest":
        res_pred = ml.RandomForest()
    elif logic_select == "LoginRression":
        res_pred = ml.LoginRegression()


    return JsonResponse({"msg":str(round(res_pred[0][0],2))+"万","acc":res_pred[1]})

