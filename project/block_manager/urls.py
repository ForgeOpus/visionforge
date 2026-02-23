from django.urls import path, include
from rest_framework.routers import DefaultRouter

from block_manager.views.project_views import ProjectViewSet
from block_manager.views.architecture_views import (
    save_architecture,
    load_architecture,
    get_node_definitions,
    get_node_definition,
    render_node_code
)
from block_manager.views.validation_views import validate_model
from block_manager.views.export_views import export_model
from block_manager.views.chat_views import chat_message, get_suggestions, get_environment_info
from block_manager.views.group_views import group_definition_list, group_definition_detail
from block_manager.views.sharing_views import (
    get_shared_project,
    get_shared_architecture,
    enable_sharing,
    disable_sharing,
)

# Create router for viewsets
router = DefaultRouter()
router.register(r'projects', ProjectViewSet, basename='project')

urlpatterns = [
    # Include router URLs
    path('', include(router.urls)),
    
    # Architecture endpoints
    path('projects/<uuid:project_id>/save-architecture', save_architecture, name='save-architecture'),
    path('projects/<uuid:project_id>/load-architecture', load_architecture, name='load-architecture'),

    # Group definition endpoints
    path('projects/<uuid:project_id>/groups', group_definition_list, name='group-definition-list'),
    path('projects/<uuid:project_id>/groups/<uuid:definition_id>', group_definition_detail, name='group-definition-detail'),

    # Node definition endpoints
    path('node-definitions', get_node_definitions, name='node-definitions'),
    path('node-definitions/<str:node_type>', get_node_definition, name='node-definition'),
    path('render-node-code', render_node_code, name='render-node-code'),
    
    # Validation endpoint (matches frontend API contract)
    path('validate', validate_model, name='validate-model'),
    
    # Export endpoint (matches frontend API contract)
    path('export', export_model, name='export-model'),
    
    # Chat endpoints (matches frontend API contract)
    path('chat', chat_message, name='chat-message'),
    path('suggestions', get_suggestions, name='suggestions'),

    # Environment info endpoint
    path('environment', get_environment_info, name='environment-info'),

    # Sharing endpoints
    path('shared/<uuid:share_token>/', get_shared_project, name='shared-project'),
    path('shared/<uuid:share_token>/architecture/', get_shared_architecture, name='shared-architecture'),
    path('projects/<uuid:project_id>/share/', enable_sharing, name='enable-sharing'),
    path('projects/<uuid:project_id>/unshare/', disable_sharing, name='disable-sharing'),
]
