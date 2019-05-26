from django.shortcuts import render
from django.shortcuts import render
from .ml import MachineLearn
from django.http import JsonResponse


# Create your views here.

def index(request):
    return render(request, "index.html")


def pred(request):
    sex = request.POST.get("sex", None)
    age = request.POST.get("age", None)
    fare = request.POST.get("fare", None)
    logic_select = request.POST.get("logic_select", None)

    ml = MachineLearn(sex, age, fare).KD()
    res_pred = ""
    if logic_select == "KNN":
        res_pred = ml[0]
    elif logic_select == "LogicRegression":
        # res_pred = ml.LogicRegression
        res_pred = ml[1]
    elif logic_select == "DecisionTree":
        # res_pred = ml.DecisionTree
        res_pred = ml[2]
    elif logic_select == "RandomForest":
        # res_pred = ml.RandomForest
        res_pred = ml[3]
    elif logic_select == "SVM":
        # res_pred = ml.SVM
        res_pred = ml[4]
    else:
        pass
    # print(str(res_pred[0]),res_pred[1])
    # print(res_pred[1])

    return JsonResponse({"msg": str(res_pred[0]), "acc": res_pred[1]})

# Create your views here.
