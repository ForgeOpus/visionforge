"""
URL configuration for backend project.

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
from django.urls import path, include, re_path
from django.views.generic import TemplateView
from django.conf import settings
from django.views.static import serve
from django.views.decorators.csrf import ensure_csrf_cookie
from django.utils.decorators import method_decorator

# Ensure CSRF cookie is set when serving the React app
@method_decorator(ensure_csrf_cookie, name='dispatch')
class ReactAppView(TemplateView):
    template_name = 'index.html'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('block_manager.urls')),
    # Serve assets directory (React build files)
    re_path(r'^assets/(?P<path>.*)$', serve, {'document_root': settings.BASE_DIR / 'frontend_build' / 'assets'}),
    # Catch-all: Serve React app for all other routes
    # Must be LAST - only matches routes not already matched above
    re_path(r'^', ReactAppView.as_view(), name='react-app'),
]
