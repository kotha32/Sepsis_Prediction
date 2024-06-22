"""fakenewsdetect URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
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
from django.urls import path
from app import views
from django.urls import include, re_path
urlpatterns = [
    re_path('admin/', admin.site.urls),
    re_path(r'^home', views.home, name='home'),
	re_path(r'^nvb', views.nvb, name='nvb'),
	re_path(r'^pac', views.pac, name='pac'),
	re_path(r'^svm', views.svm, name='svm'),
	re_path(r'^dec', views.dec, name='dec'),
	re_path(r'^randomf', views.randomf, name='randomf'),
	re_path(r'^mnb', views.mnb, name='mnb'),
	re_path(r'^graph', views.graph, name='graph'),
	re_path(r'^$', views.accuracy, name='accuracy'),
	re_path(r'^loginCheck', views.loginCheck, name='loginCheck'),
	re_path(r'^reg', views.reg, name='reg'),
	re_path(r'^login', views.login, name='login'),
	re_path(r'^save', views.save, name='save'),
	re_path(r'^logout', views.logout, name='logout'),
	re_path(r'^graph2', views.graph2, name='graph2'),
]
