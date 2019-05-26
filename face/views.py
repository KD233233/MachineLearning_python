from django.shortcuts import render
from .ml import MachineLearn
from django.http import JsonResponse,HttpResponse
# from django.http import  request
import os
from .pgm import Pgm

# Create your views here.

def index(request):
    return render(request,'index3.html')

def pred(request):
    file_path = request.POST.get('file_path',None)
    pca = int(request.POST.get('pca',None))
    ml = MachineLearn(file_path,pca).KD()
    logic_select = request.POST.get("logic_select", None)
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
    elif logic_select == "DNN_keras":
        pred = MachineLearn(file_path,pca).DNN_keras()
        # res_pred = m1()
    return JsonResponse({"msg": str(res_pred[0]), "acc": res_pred[1]})


def upload(request):
    '''返回jsonresponse({'msg':"hello","status":"world"})'''
    ret = {}
    if request.method == 'POST':
        ret = {"status":False,"data":None,"error":None}
    try:
        pca = request.POST.get('pca')
        img = request.FILES.get('img')
        FILE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep + 'static' + os.sep + img.name
        OUT_PATH = FILE_PATH + '.jpg'
        FILE_PATH_URL = '/static/' + img.name + '.jpg'
        f = open(FILE_PATH,'wb')
        for chunk in img.chunks(chunk_size=10241024):
            f.write(chunk)
        ret["status"] = True
        pgm = Pgm()
        pgm.pgm2jpg(FILE_PATH,OUT_PATH)
        ret["data"] = os.path.join("static",img.name)

    except Exception as e:
        ret['error'] = e
        return JsonResponse({"file_path":"","file_path_url":"","status":ret['status'],"error":ret['error']})
    finally:
        f.close()
    return JsonResponse({"file_path":FILE_PATH,"file_path_url":FILE_PATH_URL,"status":ret['status'],"error":ret['error']})


