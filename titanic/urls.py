#coding:gbk
from titanic.views import index,pred
from django.urls import path,include

urlpatterns = [
    path('',index,name="index"),
    path('pred/',pred,name="pred"),
]

