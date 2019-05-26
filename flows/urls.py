#coding:gbk
from flows.views import index,linear_pred,pred
from django.urls import path
urlpatterns = [
    path('',index,name="index"),
    path('linear_pred/',linear_pred,name="linear_pred"),
    path('pred/',pred,name="pred")
]

