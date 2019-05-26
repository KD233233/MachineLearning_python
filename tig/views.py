from django.shortcuts import render
from django.http import JsonResponse
from .ml import MachineLearn
import os


# Create your views here.
def index(request):
    return render(request, 'index5.html')

def pred(request):
    file_path = request.POST.get('file_path', None)
    logic_select = request.POST.get("logic_select", None)
    ml = MachineLearn(file_path)
    res_pred = ""
    if logic_select == "DNN_Keras":
        res_pred = ml.DNN_keras()
    # return JsonResponse({"msg": str(res_pred[0]), "acc": str(res_pred[1])})
    elif logic_select == "MLP_Keras":
        res_pred = ml.MLP_Keras()
    elif logic_select == "DNN_Tensorflow":
        res_pred = ml.DNN_Tensorflow()
    return JsonResponse({"msg": str(res_pred[0]), "acc": str(res_pred[1])})


def upload(request):
    ret = {}
    FILE_PATH = "'"
    if request.method == "POST":
        ret = {"status": False, "data": None, "error": None}
        try:
            img = request.FILES.get("img")
            FILE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep + "static" + os.sep + img.name
            OUT_PATH = FILE_PATH
            FILE_PATH_URL = "/static/" + img.name
            f = open(FILE_PATH, 'wb')
            for chunk in img.chunks(chunk_size=10241024):
                f.write(chunk)
            ret["status"] = True
            ret["data"] = os.path.join("static", img.name)
        except Exception as e:
            ret["error"] = e
            return JsonResponse({'file_path': "", "file_path_url": "", "status": ret["status"], "error": ret["error"]})
        finally:
            f.close()
    return JsonResponse(
        {'file_path': FILE_PATH, "file_path_url": FILE_PATH_URL, "status": ret["status"], "error": ret["error"]})
