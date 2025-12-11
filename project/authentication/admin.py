from django.contrib import admin
from .models import User


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('firebase_uid', 'email', 'display_name', 'auth_provider',
                   'tier', 'project_count', 'created_at', 'last_login_at')
    list_filter = ('auth_provider', 'tier', 'created_at')
    search_fields = ('email', 'display_name', 'firebase_uid')
    readonly_fields = ('firebase_uid', 'created_at', 'first_model_created_at',
                      'first_export_at', 'days_to_first_export')

    fieldsets = (
        ('Authentication', {
            'fields': ('firebase_uid', 'email', 'display_name', 'avatar_url', 'auth_provider')
        }),
        ('Subscription', {
            'fields': ('tier', 'project_count')
        }),
        ('Engagement', {
            'fields': ('created_at', 'last_login_at', 'total_sessions', 'total_time_spent_minutes')
        }),
        ('Conversion Funnel', {
            'fields': ('first_model_created_at', 'first_export_at', 'days_to_first_export')
        }),
    )
