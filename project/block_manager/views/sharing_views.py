import uuid
import logging
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.core.exceptions import ObjectDoesNotExist

from block_manager.models import Project

logger = logging.getLogger(__name__)


@api_view(['GET'])
def get_shared_project(request, share_token):
    """
    Public endpoint — returns project metadata for a shared project.
    No authentication required.
    Returns 404 if the token doesn't exist or sharing is disabled.
    """
    try:
        project = Project.objects.get(share_token=share_token, is_shared=True)
    except Project.DoesNotExist:
        return Response(
            {'error': 'Shared project not found or link is no longer active'},
            status=status.HTTP_404_NOT_FOUND
        )

    owner_display_name = None
    if project.user:
        owner_display_name = project.user.display_name or "Anonymous"

    return Response({
        'name': project.name,
        'description': project.description,
        'framework': project.framework,
        'owner_display_name': owner_display_name,
        'share_token': str(project.share_token),
    })


@api_view(['GET'])
def get_shared_architecture(request, share_token):
    """
    Public endpoint — returns the canvas state for a shared project.
    No authentication required.
    Returns 404 if the token doesn't exist or sharing is disabled.
    """
    try:
        project = Project.objects.get(share_token=share_token, is_shared=True)
    except Project.DoesNotExist:
        return Response(
            {'error': 'Shared project not found or link is no longer active'},
            status=status.HTTP_404_NOT_FOUND
        )

    try:
        architecture = project.architecture
    except ObjectDoesNotExist:
        return Response({'nodes': [], 'edges': [], 'groupDefinitions': []})
    except Exception:
        logger.exception('Unexpected error fetching architecture for project %s', project.id)
        return Response({'nodes': [], 'edges': [], 'groupDefinitions': []})

    if architecture.canvas_state:
        return Response(architecture.canvas_state)

    return Response({'nodes': [], 'edges': [], 'groupDefinitions': []})


@api_view(['POST'])
def enable_sharing(request, project_id):
    """
    Enable public sharing for a project.
    Authentication required; only the project owner can call this.
    Generates a share_token on first use; reuses the existing token on subsequent calls
    so the perma-link stays stable.
    """
    if not hasattr(request, 'firebase_user') or not request.firebase_user:
        return Response(
            {'error': 'Authentication required'},
            status=status.HTTP_401_UNAUTHORIZED
        )

    project = get_object_or_404(Project, pk=project_id)

    if project.user != request.firebase_user:
        return Response(
            {'error': 'You do not have permission to share this project'},
            status=status.HTTP_403_FORBIDDEN
        )

    if project.share_token is None:
        project.share_token = uuid.uuid4()

    project.is_shared = True
    project.save(update_fields=['share_token', 'is_shared'])

    return Response({
        'share_token': str(project.share_token),
        'is_shared': project.is_shared,
    })


@api_view(['DELETE'])
def disable_sharing(request, project_id):
    """
    Disable public sharing for a project.
    Authentication required; only the project owner can call this.
    The share_token is preserved so re-enabling restores the same URL.
    """
    if not hasattr(request, 'firebase_user') or not request.firebase_user:
        return Response(
            {'error': 'Authentication required'},
            status=status.HTTP_401_UNAUTHORIZED
        )

    project = get_object_or_404(Project, pk=project_id)

    if project.user != request.firebase_user:
        return Response(
            {'error': 'You do not have permission to modify this project'},
            status=status.HTTP_403_FORBIDDEN
        )

    project.is_shared = False
    project.save(update_fields=['is_shared'])

    return Response({'is_shared': False})
