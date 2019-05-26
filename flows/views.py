from django.shortcuts import render
from .ml import MachineLearn
from django.http import JsonResponse,HttpResponse

# Create your views here.

def index(request):
    return render(request,'index2.html')

def linear_pred(request):
    petal_w = float(request.POST.get("petal_width", None))
    linear_select = request.POST.get("linear_select", None)

    m = ''
    if linear_select == "LinearRegression":
        m = MachineLearn()
    petal_length = m.Linear(petal_w)

    # return JsonResponse({"msg": str(res_pred[0]), "acc": res_pred[1]})
    return JsonResponse({"msg": petal_length[0][0]})


def pred(request):
    petal_w2 = float(request.POST.get("petal_width", None))
    petal_length = float(request.POST.get("petal_length", None))
    calyx_w = float(request.POST.get("sepal_width", None))
    calyx_h = float(request.POST.get("sepal_length", None))
    logic_select = request.POST.get("logic_select", None)

    ml = MachineLearn().KD(a=[[petal_w2,petal_length,calyx_w,calyx_h]])

    res_pred = ""
    if logic_select == "KNN":
        res_pred = ml[0]
    elif logic_select == "LogicRegression":
        res_pred = ml[1]
    elif logic_select == "DecisionTree":
        res_pred = ml[2]
    elif logic_select == "RandomForest":
        res_pred = ml[3]
    elif logic_select == "SVM":
        res_pred = ml[4]
    elif logic_select == "KMeans":
        m = MachineLearn()
        res_pred = m.KMeans(pred=[[petal_w2,petal_length,calyx_w,calyx_h]])
    else:
        pass
    # print(str(res_pred[0]),res_pred[1])
    # print(res_pred[1])
    print(str(res_pred[0][0]),res_pred[1])

    return JsonResponse({"msg": str(res_pred[0][0]), "acc": str(res_pred[1])})

