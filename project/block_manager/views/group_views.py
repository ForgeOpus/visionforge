from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from block_manager.models import Project, GroupBlockDefinition
from block_manager.serializers import GroupBlockDefinitionSerializer


@api_view(['GET', 'POST'])
def group_definition_list(request, project_id):
    """
    List all group definitions for a project or create a new one.
    """
    project = get_object_or_404(Project, id=project_id)

    if request.method == 'GET':
        definitions = GroupBlockDefinition.objects.filter(project=project)
        serializer = GroupBlockDefinitionSerializer(definitions, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = GroupBlockDefinitionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(project=project)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PUT', 'DELETE'])
def group_definition_detail(request, project_id, definition_id):
    """
    Retrieve, update, or delete a group definition.
    """
    project = get_object_or_404(Project, id=project_id)
    definition = get_object_or_404(GroupBlockDefinition, id=definition_id, project=project)

    if request.method == 'GET':
        serializer = GroupBlockDefinitionSerializer(definition)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = GroupBlockDefinitionSerializer(definition, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        # Check if there are instances using this definition
        instances_count = definition.instances.count()

        # Get cascade option from query params (default: False)
        cascade = request.query_params.get('cascade', 'false').lower() == 'true'

        if instances_count > 0 and not cascade:
            return Response(
                {
                    'error': 'Cannot delete definition with active instances',
                    'instances_count': instances_count,
                    'message': f'{instances_count} block instance(s) are using this definition. Use cascade=true to delete all instances.'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Delete the definition (CASCADE will handle instances if cascade=true)
        definition.delete()
        return Response(
            {'message': 'Group definition deleted successfully'},
            status=status.HTTP_204_NO_CONTENT
        )
