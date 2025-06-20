"""
URL configuration for anwendungsprojekt project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
from football_prediction.views import team_selection_view
from django.urls import path
from football_prediction import views
from django.urls import path, include

urlpatterns = [
    path('', views.football_prediction, name='home'),
    path('admin/', admin.site.urls),
    path('overview/', views.model_overview, name='overview'),
    path('prediction/', views.team_selection_view, name='team_selection'),  # ← Name ergänzt
    path('dashboard/', views.dashboard_view, name='dashboard'),
]
