from . import views
from django.urls import path

app_name="rent"

urlpatterns = [
    path('',views.index,name="index"),
    path('pred/',views.pred,name="pred"),
]

