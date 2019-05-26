#coding:gbk
from tig.views import index,pred,upload

from django.urls import path
urlpatterns = [
    path('',index,name="index"),
    path('pred/',pred,name="pred"),
    path('upload/',upload,name='upload')
]

