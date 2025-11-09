from django.contrib import admin
from .models import Project, ModelArchitecture, Block, Connection


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['name', 'framework', 'created_at', 'updated_at']
    list_filter = ['framework', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(ModelArchitecture)
class ModelArchitectureAdmin(admin.ModelAdmin):
    list_display = ['project', 'is_valid', 'created_at', 'updated_at']
    list_filter = ['is_valid', 'created_at']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(Block)
class BlockAdmin(admin.ModelAdmin):
    list_display = ['node_id', 'block_type', 'architecture', 'created_at']
    list_filter = ['block_type', 'created_at']
    search_fields = ['node_id', 'block_type']
    readonly_fields = ['id', 'created_at']


@admin.register(Connection)
class ConnectionAdmin(admin.ModelAdmin):
    list_display = ['edge_id', 'source_block', 'target_block', 'is_valid', 'created_at']
    list_filter = ['is_valid', 'created_at']
    search_fields = ['edge_id']
    readonly_fields = ['id', 'created_at']
