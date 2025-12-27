"""
URL routing for maintenance endpoints.
"""
from django.urls import path
from block_manager.views import maintenance_views

urlpatterns = [
    path('cleanup-files', maintenance_views.trigger_file_cleanup, name='trigger_file_cleanup'),
    path('upload-stats', maintenance_views.get_upload_stats, name='upload_stats'),
]
