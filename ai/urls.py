"""ai URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from titanic.views import index, pred

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('',index,name="index"),
    # path('pred/',pred,name="pred"),
    # path('', include('titanic.urls')),
    # path('', include('flows.urls')),
    # path('', include('face.urls')),
    # path('', include('animal.urls')),
    # path('', include('tig.urls')),
    path('', include('rent.urls')),
    # path('', include('pre.urls')),
]
