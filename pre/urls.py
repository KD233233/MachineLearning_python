# coding:gbk
from . import views
from django.urls import path, include

app_name = "quant"
urlpatterns = [
    path('', views.index, name="index"),
    path('pred/', views.pred, name="pred"),
    path("back/", views.back, name="back"),
    path("setcode/", views.setcode, name="setcode")

]
