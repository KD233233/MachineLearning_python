from django.shortcuts import render
from .ml import CatDog
from django.http import JsonResponse,HttpResponse
# from django.http import  request
from .pgm import Pgm
import os

# Create your views here.

def index(request):
    return render(request, 'index4.html')


def pred(request):
    file_path = request.POST.get("file_path",None)
    cat_dog = CatDog(file_path)
    cat_dog.ini_model()
    cat_dog.train_cat_dog()
    res_pred = cat_dog.pred_one_cat_dog()
    cat_dog.save_my_model()
    cat_dog.load_my_model()

    return JsonResponse({"msg":res_pred})


def upload(request):
    '''返回jsonresponse({'msg':"hello","status":"world"})'''
    ret = {}
    if request.method == 'POST':
        ret = {"status": False, "data": None, "error": None}
    try:
        pca = request.POST.get('pca')
        img = request.FILES.get('img')
        FILE_PATH = os.path.abspath(os.path.dirname(__file__)) + os.sep + 'static' + os.sep + img.name
        OUT_PATH = FILE_PATH + '.jpg'
        FILE_PATH_URL = '/static/' + img.name
        f = open(FILE_PATH, 'wb')
        for chunk in img.chunks(chunk_size=10241024):
            f.write(chunk)
        ret["status"] = True
        # pgm = Pgm()
        # pgm.pgm2jpg(FILE_PATH, OUT_PATH)
        ret["data"] = os.path.join("static", img.name)

    except Exception as e:
        ret['error'] = e
        return JsonResponse({"file_path": "", "file_path_url": "", "status": ret['status'], "error": ret['error']})
    finally:
        f.close()
    return JsonResponse(
        {"file_path": FILE_PATH, "file_path_url": FILE_PATH_URL, "status": ret['status'], "error": ret['error']})


