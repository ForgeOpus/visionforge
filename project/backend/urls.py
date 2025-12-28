"""
URL configuration for backend project.
Serves API endpoints and React SPA
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static
from backend.health import health_check

urlpatterns = [
    # Admin interface
    path('admin/', admin.site.urls),

    # Health check
    path('health/', health_check, name='health_check'),

    # API endpoints - these must come BEFORE the catch-all route
    path('api/v1/', include('block_manager.urls')),
    path('api/v1/auth/', include('authentication.urls')),
    path('api/v1/maintenance/', include('block_manager.maintenance_urls')),
]

# Serve static files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# Catch-all route for React SPA (must be last)
# This serves index.html for all non-API routes to support React Router
urlpatterns += [
    re_path(r'^.*$', TemplateView.as_view(template_name='index.html'), name='frontend'),
]
