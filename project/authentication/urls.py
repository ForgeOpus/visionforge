"""
URL routing for authentication API endpoints.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('csrf', views.get_csrf_token, name='csrf_token'),
    path('verify-token', views.verify_token, name='verify_token'),
    path('me', views.get_current_user, name='get_current_user'),
    path('update-session', views.update_session, name='update_session'),
    path('logout', views.logout, name='logout'),
    path('milestone', views.mark_milestone, name='mark_milestone'),
]
